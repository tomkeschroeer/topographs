from tracemalloc import take_snapshot

import numpy as np
import numpy.ma as ma


def calc_delta_R_squared(reco_infos, truth_infos):
    if np.isnan(reco_infos).any() or np.isnan(truth_infos).any():
        return np.nan
    return (reco_infos[0] - truth_infos[0]) ** 2 + (reco_infos[1] - truth_infos[1]) ** 2


def calc_delta_R_squared_matrix(reco_infos, truth_infos):
    return np.frompyfunc(calc_delta_R_squared, 2, 1).outer(reco_infos, truth_infos)


def calc_phi_jet(phi_truth, dR_truth, eta_truth, eta_jet):
    """
    calculates the phi value for the jet
    """
    if np.isnan([phi_truth, dR_truth, eta_truth, eta_jet]).any():
        return np.nan
    return phi_truth - np.sqrt(dR_truth**2 - (eta_truth - eta_jet) ** 2)


def calc_phi_jet_array(phi_truth, dR_truth, eta_truth, eta_jet):
    return map(calc_phi_jet, phi_truth, dR_truth, eta_truth, eta_jet)


def calc_phi_reco(phi_jet, dR_reco, eta_reco, eta_jet):
    """
    calculates the phi value for the reconstructed track
    """
    if np.isnan([phi_jet, dR_reco, eta_reco, eta_jet]).any():
        return np.nan
    return dR_reco**2 - (eta_reco - eta_jet) ** 2 + phi_jet


def calc_phi_reco_array(phi_jet, dR_reco, eta_reco, eta_jet):
    return map(calc_phi_reco, phi_jet, dR_reco, eta_reco, eta_jet)


class Matcher:
    def __init__(
        self,
        dR_truth: np.array,
        dR_reco: np.array,
        phi_truth: np.array,
        eta_truth: np.array,
        eta_jet: np.array,
        eta_reco: np.array,
    ):
        self.dR_truth = ma.masked_equal(dR_truth, np.nan)
        self.dR_reco = ma.masked_equal(dR_reco, np.nan)
        self.phi_truth = ma.masked_equal(phi_truth, np.nan)
        self.eta_truth = ma.masked_equal(eta_truth, np.nan)
        self.eta_jet = ma.masked_equal(eta_jet, np.nan)
        self.eta_reco = ma.masked_equal(eta_reco, np.nan)

        self.phi_jet = list(
            calc_phi_jet_array(
                self.phi_truth, self.dR_truth, self.eta_truth, self.eta_jet
            )
        )
        self.phi_reco = list(
            calc_phi_reco_array(self.phi_jet, self.dR_reco, self.eta_reco, self.eta_jet)
        )

        self.truth_infos = np.empty(
            len(self.eta_truth), dtype=[("phi_truth", float), ("eta_truth", float)]
        )
        self.truth_infos["phi_truth"] = self.phi_truth
        self.truth_infos["eta_truth"] = self.eta_truth
        self.reco_infos = np.empty(
            len(self.eta_reco), dtype=[("phi_reco", float), ("eta_reco", float)]
        )
        self.reco_infos["phi_reco"] = self.phi_reco
        self.reco_infos["eta_reco"] = self.eta_reco

    def matching_truth_tracks(self):
        """ """
        DeltaR_matrix = ma.masked_equal(
            calc_delta_R_squared_matrix(self.truth_infos, self.reco_infos), np.nan
        )
        print(DeltaR_matrix)
