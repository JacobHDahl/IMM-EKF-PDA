"""

"""
# %% Imports

# types
from typing import (
    Tuple,
    List,
    TypeVar,
    Optional,
    Dict,
    Any,
    Union,
    Sequence,
    Generic,
    Iterable,
)
from mixturedata import MixtureParameters
from gaussparams import GaussParams
from estimatorduck import StateEstimator

# packages
from dataclasses import dataclass

# from singledispatchmethod import singledispatchmethod
import numpy as np
from scipy import linalg
from scipy.special import logsumexp

# local
import discretebayes

# %% TypeVar and aliases
MT = TypeVar("MT")  # a type variable to be the mode type

# %% IMM
@dataclass
class IMM(Generic[MT]):
    # The M filters the IMM relies on
    filters: List[StateEstimator[MT]]
    # the transition matrix. PI[i, j] = probability of going from model i to j: shape (M, M)
    PI: np.ndarray

    def __post_init__(self):
        assert (
            self.PI.ndim == 2
        ), "Transition matrix PI shape must be (len(filters), len(filters))"
        assert (
            self.PI.shape[0] == self.PI.shape[1]
        ), "Transition matrix PI shape must be (len(filters), len(filters))"
        assert np.allclose(
            self.PI.sum(axis=1), 1
        ), "The rows of the transition matrix PI must sum to 1."

        assert (
            len(self.filters) == self.PI.shape[0]
        ), "Transition matrix PI shape must be (len(filters), len(filters))"

    def mix_probabilities(
        self,
        immstate: MixtureParameters[MT],
        # sampling time
        Ts: float,
    ) -> Tuple[
        np.ndarray, np.ndarray
    ]:  # predicted_mode_probabilities, mix_probabilities: shapes = ((M, (M ,M))).
        # mix_probabilities[s] is the mixture weights for mode s
        """Calculate the predicted mode probability and the mixing probabilities."""

        predicted_mode_probabilities, mix_probabilities = discretebayes.discrete_bayes(
            immstate.weights, self.PI
        )

        assert predicted_mode_probabilities.shape == (
            self.PI.shape[0],
        ), "IMM.mix_probabilities: Wrong shape on the predicted mode probabilities"
        assert (
            mix_probabilities.shape == self.PI.shape
        ), "IMM.mix_probabilities: Wrong shape on mixing probabilities"
        assert np.all(
            np.isfinite(predicted_mode_probabilities)
        ), "IMM.mix_probabilities: predicted mode probabilities not finite"
        assert np.all(
            np.isfinite(mix_probabilities)
        ), "IMM.mix_probabilities: mix probabilities not finite"
        assert np.allclose(
            mix_probabilities.sum(axis=1), 1
        ), "IMM.mix_probabilities: mix probabilities does not sum to 1 per mode"

        return predicted_mode_probabilities, mix_probabilities

    def mix_states(
        self,
        immstate: MixtureParameters[MT],
        # the mixing probabilities: shape=(M, M)
        mix_probabilities: np.ndarray,
    ) -> List[MT]:
        mixed_states = [
            fs.reduce_mixture(MixtureParameters(mix_pr_s, immstate.components))
            for fs, mix_pr_s in zip(self.filters, mix_probabilities)
        ]
        return mixed_states

    def mode_matched_prediction(
        self,
        mode_states: List[MT],
        # The sampling time
        Ts: float,
    ) -> List[MT]:
        modestates_pred = [
            fs.predict(cs, Ts) for fs, cs in zip(self.filters, mode_states)
        ]
        return modestates_pred

    def predict(
        self,
        immstate: MixtureParameters[MT],
        # sampling time
        Ts: float,
    ) -> MixtureParameters[MT]:
        """
        Predict the immstate Ts time units ahead approximating the mixture step.

        Ie. Predict mode probabilities, condition states on predicted mode,
        appoximate resulting state distribution as Gaussian for each mode, then predict each mode.
        """

        predicted_mode_probability, mixing_probability = self.mix_probabilities(
            immstate, Ts
        )

        mixed_mode_states: List[MT] = self.mix_states(immstate, mixing_probability)

        predicted_mode_states = self.mode_matched_prediction(mixed_mode_states, Ts)

        predicted_immstate = MixtureParameters(
            predicted_mode_probability, predicted_mode_states
        )
        return predicted_immstate

    def mode_matched_update(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> List[MT]:
        """Update each mode in immstate with z in sensor_state."""

        updated_state = [
            fs.update(z, cs, sensor_state=sensor_state)
            for fs, cs in zip(self.filters, immstate.components)
        ]

        return updated_state

    def update_mode_probabilities(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Dict[str, Any] = None,
    ) -> np.ndarray:
        """Calculate the mode probabilities in immstate updated with z in sensor_state"""

        loglikelihood = np.array(
            [
                fs.loglikelihood(z, cs, sensor_state=sensor_state)
                for fs, cs in zip(self.filters, immstate.components)
            ]
        )
        weight_log = np.log(immstate.weights)
        logjoint = loglikelihood + weight_log

        updated_mode_probabilities = np.exp(logjoint - logsumexp(logjoint))

        assert np.all(
            np.isfinite(updated_mode_probabilities)
        ), "IMM.update_mode_probabilities: updated probabilities not finite "
        assert np.allclose(
            np.sum(updated_mode_probabilities), 1
        ), "IMM.update_mode_probabilities: updated probabilities does not sum to one"

        return updated_mode_probabilities

    def update(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Dict[str, Any] = None,
    ) -> MixtureParameters[MT]:
        """Update the immstate with z in sensor_state."""

        updated_weights = self.update_mode_probabilities(
            z, immstate, sensor_state=sensor_state
        )
        updated_states = self.mode_matched_update(
            z, immstate, sensor_state=sensor_state
        )

        updated_immstate = MixtureParameters(updated_weights, updated_states)
        return updated_immstate

    def step(
        self,
        z,
        immstate: MixtureParameters[MT],
        Ts: float,
        sensor_state: Dict[str, Any] = None,
    ) -> MixtureParameters[MT]:
        """Predict immstate with Ts time units followed by updating it with z in sensor_state"""

        predicted_immstate = self.predict(immstate, Ts)
        updated_immstate = self.update(z, predicted_immstate, sensor_state=sensor_state)

        return updated_immstate

    def loglikelihood(
        self,
        z: np.ndarray,
        immstate: MixtureParameters,
        *,
        sensor_state: Dict[str, Any] = None,
    ) -> float:

        mode_conditioned_ll = np.fromiter(
            (
                fs.loglikelihood(z, modestate_s, sensor_state = sensor_state)  
                for fs, modestate_s in zip(self.filters, immstate.components)
            ),
            dtype=float,
        )

        ll = logsumexp(mode_conditioned_ll, b = immstate.weights) #needs b=

        assert np.isfinite(ll), "IMM.loglikelihood: ll not finite"
        assert isinstance(ll, float) or isinstance(
            ll.item(), float
        ), "IMM.loglikelihood: did not calculate ll to be a single float"
        return ll

    def reduce_mixture(
        self, immstate_mixture: MixtureParameters[MixtureParameters[MT]]
    ) -> MixtureParameters[MT]:

        weights = immstate_mixture.weights  
        component_conditioned_mode_prob = np.array(
            [c.weights.ravel() for c in immstate_mixture.components]
        )

        mode_prob, mode_conditioned_component_prob = discretebayes.discrete_bayes(weights, component_conditioned_mode_prob)

        comps_per_mode = zip(*[comp.components for comp in immstate_mixture.components])
        mode_states = [
            fs.reduce_mixture(MixtureParameters(mode_s_cond_comp_prob, mode_s_comp))
            for fs, mode_s_cond_comp_prob, mode_s_comp in zip(
                self.filters, mode_conditioned_component_prob, comps_per_mode,
            )
        ]

        return MixtureParameters(mode_prob, mode_states)

    def estimate(self, immstate: MixtureParameters[MT]) -> GaussParams:
        dataRed = self.filters[0].reduce_mixture(immstate)
        return self.filters[0].estimate(dataRed)

    def gate(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        gate_size_square: float,
        sensor_state: Dict[str, Any] = None,
    ) -> bool:
        mode_gated: List[bool] = []

        for mode in immstate.components:
            mode_gated.append( self.filters[0].gate(z, mode, gate_size_square, sensor_state=sensor_state))

        return np.any(mode_gated)

    def NISes(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, np.ndarray]:
        """Calculate NIS per mode and the average"""
        NISes = np.array(
            [
                fs.NIS(z, ms, sensor_state=sensor_state)
                for fs, ms in zip(self.filters, immstate.components)
            ]
        )

        innovs = [
            fs.innovation(z, ms, sensor_state=sensor_state)
            for fs, ms in zip(self.filters, immstate.components)
        ]

        v_ave = np.average([gp.mean for gp in innovs], axis=0, weights=immstate.weights)
        S_ave = np.average([gp.cov for gp in innovs], axis=0, weights=immstate.weights)

        NIS = (v_ave * np.linalg.solve(S_ave, v_ave)).sum()
        return NIS, NISes

