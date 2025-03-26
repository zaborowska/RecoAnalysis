import ROOT
import argparse
from math import sqrt
import numpy
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate

parser = argparse.ArgumentParser(description='Analyse calo shower data for SINGLE particle gun events')
parser.add_argument('--infile', '-i', required=True, type=str, nargs='+', help='EDM4hep file to analyse with particle gun events. Output of reco with Pandora, e.g. CLDConfig')
parser.add_argument('-o', '--outfile', type=str, default="showerAnalysis.root", help='output file')
parser.add_argument('-n', '--ncpus', type=int, default=2, help='Number of CPUs to use in analysis')
parser.add_argument('--trueE', action='store_true', help='Fit e res around true particle energy taken a the first MC particle in the list of particles, a way to go if the default hist mean fails')
parser.add_argument('--interactive', action='store_true', help='Pause to inspect a canvas')
parser.add_argument('--verbose', action='store_true', help='Verbose output for debugging')
args = parser.parse_args()

ROOT.gSystem.Load("libedm4hep")
ROOT.gInterpreter.Declare("#include <edm4hep/SimCalorimeterHitData.h>")
ROOT.gInterpreter.Declare("#include <edm4hep/CalorimeterHitData.h>")
ROOT.gInterpreter.Declare("#include <edm4hep/MCParticleData.h>")
ROOT.gInterpreter.Declare("#include <edm4hep/ReconstructedParticleData.h>")
ROOT.gInterpreter.Declare("#include <edm4hep/RecoMCParticleLinkCollection.h>")
ROOT.gInterpreter.Declare("#include <edm4hep/CaloHitMCParticleLinkCollection.h>")
ROOT.gInterpreter.Declare("#include <edm4hep/ClusterData.h>")
ROOT.gInterpreter.Declare("#include <edm4hep/RecoMCParticleLinkData.h>")
ROOT.gStyle.SetOptStat(0000)
#__________________________________________________________
def run(inputlist, outname, ncpu):
    outname = outname
    ROOT.ROOT.EnableImplicitMT(ncpu)
    df = ROOT.RDataFrame("events", inputlist)
    print ("Initialization done")

    ## Declare a function that returns links (reco-MC) as IDs
    ROOT.gInterpreter.Declare("""
    // Get a vector of pairs linking reconstructed particle (pair.first) to an MC particle (pair.second)
    // For each of the reconstructed particle a single highest weight link is chosen (single MC)
    // However multiple reco particles may be linked to a single MC.
    // Weights are assumed to come from PandoraPFA (1e5*calo_weight+track_weight).
    // Based on reco PDG, either track_weight is taken (charged), or calo_weight (neutrals). The list of checked neutrals is not complete!
    // Links with weight below 50% are not considered. This threshold may be tuned.
    ROOT::VecOps::RVec<std::pair<int,int>> getMapReco2McFromClusters(ROOT::VecOps::RVec<int> id_reco,
                                             ROOT::VecOps::RVec<int> id_mc,
                                             ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco,
                                             ROOT::VecOps::RVec<edm4hep::MCParticleData> mc,
                                             ROOT::VecOps::RVec<edm4hep::RecoMCParticleLinkData> links) {
    std::vector<bool> linked_reco;
    linked_reco.assign(reco.size(), false);

    std::map<int,std::vector<std::pair<int,float>>> map_reco_all;
    ROOT::VecOps::RVec<std::pair<int,int>> map_reco;

    for (size_t i=0; i<links.size();i++) {
      map_reco_all[id_reco[i]].emplace_back(std::make_pair(id_mc[i], links[i].weight));
      linked_reco[id_reco[i]] = true;
    }
    std::vector<int> neutrals{22,2112,130,311};
    for(auto recPart: map_reco_all) {
      int id_maxWeight = recPart.second[0].first;
      int maxWeight = -1;
      bool ifNeutral = false;
      // decide which weight to take based on PDG of reco particle (charged/neutral)
      int recoPDG = reco[recPart.first].PDG;
      if (find(neutrals.begin(), neutrals.end(), recoPDG) != neutrals.end()) {
         ifNeutral = true;
      }
      for(auto v: recPart.second) {
        auto m = mc[v.first];
        // RecoMCTruthLinker defines Pandora weight as  weight = 10000*calo weight+track weight (weights in permill)
        // https://github.com/iLCSoft/MarlinReco/blob/eb15b0d7a864fed217327fb846cec600073a60ff/Analysis/RecoMCTruthLink/src/RecoMCTruthLinker.cc#L202
        int current_weight = -1;
        if (ifNeutral)
            current_weight = int(v.second / 10000); // calo weight
        else
            current_weight = int(v.second) % 10000; // track_weight
        if(current_weight > maxWeight) {
          maxWeight = current_weight;
          id_maxWeight = v.first;
        }
      }
      // Apply threshold on weight in units of permile
      if(maxWeight > 500)
         map_reco.push_back(std::make_pair(recPart.first,id_maxWeight));
    }
    return map_reco;
    }""")
    ROOT.gInterpreter.Declare("""
    float getEnergy(edm4hep::MCParticleData m) {
       return sqrt(m.momentum.x*m.momentum.x+m.momentum.y*m.momentum.y+m.momentum.z*m.momentum.z+m.mass*m.mass);
    }""")

    ## Define links between Reco and MC particles
    ### Unlinked Reco particles are not here! TODO handle them as well
    df_reco2mc_links = df.Define("RecoIDLinked","ROOT::VecOps::RVec<int> result; for(auto l: _MCTruthRecoLink_from) {result.emplace_back(l.index);} return result;")\
                         .Define("MCIDLinked","ROOT::VecOps::RVec<int> result; for(auto l: _MCTruthRecoLink_to) {result.emplace_back(l.index);} return result;")\
                         .Define("RecoMCLinkWeights","ROOT::VecOps::RVec<float> result; for(auto l: MCTruthRecoLink) {result.emplace_back(l.weight);} return result;")\
                         .Define("RecoIDLinkedSize","RecoIDLinked.size()")\
                         .Define("MCIDLinkedSize","MCIDLinked.size()")\
                         .Define("RecoSize","PandoraPFOs.size()")\
                         .Define("MCSize","MCParticles.size()")\
                         .Define("MCPDG", "ROOT::VecOps::RVec<int> result; for(auto& m:MCParticles){result.push_back(m.PDG);} return result;")\
                         .Define("MCEnergy", "ROOT::VecOps::RVec<float> result; for(auto& m:MCParticles){result.push_back(getEnergy(m));} return result;")\
                         .Define("recoPDG", "ROOT::VecOps::RVec<int> result; for(auto& m:PandoraPFOs){result.push_back(m.PDG);} return result;")\
                         .Define("recoEnergy", "ROOT::VecOps::RVec<float> result; for(auto& m:PandoraPFOs){result.push_back(m.energy);} return result;")\
                         .Define("recoMCpairs","getMapReco2McFromClusters(RecoIDLinked, MCIDLinked, PandoraPFOs, MCParticles, MCTruthRecoLink)")\
                         .Define("recoMCpairs_size","recoMCpairs.size()")
    ### Define histograms of Delta (reco - MC)
    h_diffE = df_reco2mc_links\
        .Define("recoMCpairs_diffE", "ROOT::VecOps::RVec<float> result; for (auto& p:recoMCpairs) result.push_back(PandoraPFOs[p.first].energy - getEnergy(MCParticles[p.second])); return result;")\
        .Histo1D("recoMCpairs_diffE")
    h_diffTheta = df_reco2mc_links\
        .Define("recoMCpairs_diffTheta", "ROOT::VecOps::RVec<float> result; for (auto& p:recoMCpairs) result.push_back(TVector3(PandoraPFOs[p.first].momentum.x,PandoraPFOs[p.first].momentum.y,PandoraPFOs[p.first].momentum.z).Theta() - TVector3(MCParticles[p.second].momentum.x,MCParticles[p.second].momentum.y,MCParticles[p.second].momentum.z).Theta()); return result;")\
        .Histo1D("recoMCpairs_diffTheta")
    h_diffPhi = df_reco2mc_links\
        .Define("recoMCpairs_diffPhi", "ROOT::VecOps::RVec<float> result; for (auto& p:recoMCpairs) result.push_back(TVector3(PandoraPFOs[p.first].momentum.x,PandoraPFOs[p.first].momentum.y,PandoraPFOs[p.first].momentum.z).Phi() - TVector3(MCParticles[p.second].momentum.x,MCParticles[p.second].momentum.y,MCParticles[p.second].momentum.z).Phi()); return result;")\
        .Histo1D("recoMCpairs_diffPhi")
    ### Energy ratio
    df_recoRatioE = df_reco2mc_links\
        .Define("recoMCpairs_ratioE", "ROOT::VecOps::RVec<float> result; for (auto& p:recoMCpairs) result.push_back(PandoraPFOs[p.first].energy / getEnergy(MCParticles[p.second])); return result;")
    h_recoRatioE = df_recoRatioE.Histo1D(("recoEratio", "E_{rec} / E_{MC}", 128,0.5,1.5),"recoMCpairs_ratioE")
    h_recoRatioE.Scale(1/h_recoRatioE.Integral())
    print("Done cluster links and basic histograms")

    ## Fit Gaussian to ratio energy histogram
    # Used to count correctly reconstructed particles
    # For pgun events it can give meaningful resolution
    f_recoPrefit = ROOT.TF1("firstRecoGaus","gaus", h_recoRatioE.GetMean() - 3. * h_recoRatioE.GetRMS(), h_recoRatioE.GetMean() + 3. * h_recoRatioE.GetRMS())
    recoResult_pre = h_recoRatioE.Fit(f_recoPrefit, "SRNQ")
    f_recoFit = ROOT.TF1("finalRecoGaus", "gaus", recoResult_pre.Get().Parameter(1) - 3. * recoResult_pre.Get().Parameter(2), recoResult_pre.Get().Parameter(1) + 3. * recoResult_pre.Get().Parameter(2) )
    recoResult = h_recoRatioE.Fit(f_recoFit, "SRNQ")
    recoResult_mean = recoResult.Get().Parameter(1)
    recoResult_meanError = recoResult.Get().Error(1)
    recoResult_resolution = recoResult.Get().Parameter(2) / recoResult.Get().Parameter(1)
    tmp_resolutionErrorSigma = recoResult.Get().Error(2) / recoResult.Get().Parameter(1)
    tmp_resolutionErrorMean = recoResult.Get().Error(1) * recoResult.Get().Parameter(2) / ( recoResult.Get().Parameter(1) ** 2)
    recoResult_resolutionError = sqrt( tmp_resolutionErrorSigma ** 2 +  tmp_resolutionErrorMean ** 2 )
    print("All PFOs: E_PFO / E_MC - fitting Gaussian...")
    print(f"\tmean energy <E>= {recoResult_mean}\t +- {recoResult_meanError}")
    print(f"\tresolution sigma(E)= {recoResult_resolution}\t +- {recoResult_resolutionError}")
    print("Done fits/")
    filterEnergy_minThreshold = recoResult_mean - 1 * recoResult_resolution * recoResult_mean
    filterEnergy_maxThreshold = recoResult_mean + 1 * recoResult_resolution * recoResult_mean
    print(f"Filtered energy ratio from {filterEnergy_minThreshold} to {filterEnergy_maxThreshold}.")

    # Multiplicities
    ROOT.gInterpreter.Declare("""
    ROOT::VecOps::RVec<int> getNumCells(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco,
                                        ROOT::VecOps::RVec<edm4hep::ClusterData> clusters,
                                        ROOT::VecOps::RVec<podio::ObjectID> cells) {
       ROOT::VecOps::RVec<int> result;
       // For each of the reco particles
       for (size_t i=0; i<reco.size();i++) {
         // Sum over all clusters of that particle
         float numCellsPerParticle = 0;
         for (size_t j = reco[i].clusters_begin; j < reco[i].clusters_end; j++) {
           numCellsPerParticle += (clusters[j].hits_end - clusters[j].hits_begin);
         }
         result.push_back(numCellsPerParticle);
       }
       return result;
    }""")
    df_clusters = df\
        .Define("recoNumClusters", "ROOT::VecOps::RVec<int> result; for(auto& p:PandoraPFOs){result.push_back(p.clusters_end - p.clusters_begin);} return result;")
    h_numClusters = df_clusters.Histo1D(("numRecoClusters", "Number of clusters per reco particle; Number of clusters; Entries", 16, -0.5, 15.5),"recoNumClusters")
    df_cells = df.Define("recoNumCells", "getNumCells(PandoraPFOs, PandoraClusters, _PandoraClusters_hits)")
    h_numCells = df_cells.Histo1D(("numRecoCells", "Number of cells per reco particle; Number of cells; Entries", 512, -0.5, 4095.5),"recoNumCells")
    h_numMC = df\
        .Define("MCEnergyAbove100MeV", "ROOT::VecOps::RVec<float> result; for(auto& m:MCParticles){auto e = getEnergy(m); if(e > 0.1) result.push_back(e);} return result;")\
        .Define("numMC","return MCEnergyAbove100MeV.size()")\
        .Histo1D(("numMC", "Particle multiplicity; Multiplicity (>100 MeV); Entries", 32, -0.5, 31.5),"numMC")
    h_numReco = df\
        .Define("RecoEnergyAbove100MeV", "ROOT::VecOps::RVec<float> result; for(auto& p:PandoraPFOs){if(p.energy > 0.1) result.push_back(p.energy);} return result;")\
        .Define("numReco","return RecoEnergyAbove100MeV.size()")\
        .Define("RecoEnergyWithinFilter", f"ROOT::VecOps::RVec<float> result; for(auto& e:RecoEnergyAbove100MeV){{ if(e > {filterEnergy_minThreshold} && e < {filterEnergy_maxThreshold}) result.push_back(e);}} return result;")\
        .Define("numReco_filteredE","return RecoEnergyWithinFilter.size()")\
        .Histo1D(("numReco", "Particle multiplicity; Multiplicity (>100 MeV); Entries", 32, -0.5, 31.5),"numReco")
    h_numClusterLinks = df_reco2mc_links.Histo1D(("numRecoClusterLinks", "Particle multiplicity; Multiplicity (cluster links); Entries", 32, -0.5, 31.5),"recoMCpairs_size")
    h_numClusterLinks_filtered = df_recoRatioE\
        .Define("linkedEnergyRatioWithinFilter", f"ROOT::VecOps::RVec<float> result; for(auto& e:recoMCpairs_ratioE){{ if(e > {filterEnergy_minThreshold} && e < {filterEnergy_maxThreshold}) result.push_back(e);}} return result;")\
        .Define("numLinks_filtered","return linkedEnergyRatioWithinFilter.size()")\
        .Histo1D(("numLinks_filtered", "Particle multiplicity; Multiplicity (filtered E); Entries", 32, -0.5, 31.5),"numLinks_filtered")
    # TODO check also number of links from cells
    print("Done multiplicities. Starting with confusion matrices.")

    ## Confusion matrix and the projections
    map_pdgs = {13:0,-13:0,
                22:1,
                11:2,-11:2,
                2112:3,130:3,
                2212:4,211:4,-211:4}
    map_labels = {0: "#mu", 1: "#gamma", 2: "e", 3: "NH", 4: "CH", 5: "other"}
    h_confusionEnergy = ROOT.TH3F("h_confusionEnergy", "PID confusion;Generated as; Reconstructed as;E (GeV)", 6, -0.5,5.5, 6, -0.5, 5.5,32, 0, 128)
    h_confusionCosTheta = ROOT.TH3F("h_confusionCosTheta", "PID confusion vs cos(#theta); Generated as; Reconstructed as;cos(#theta)", 6, -0.5,5.5, 6, -0.5, 5.5, 30, -1, 1)
    h_efficiencyEnergy = ROOT.TEfficiency("g_efficiencyEnergy", "PID confusion efficiency; E(GeV); Efficiency", 32, 0, 128)
    h_efficiencyCosTheta = ROOT.TEfficiency("g_efficiencyCosTheta", "PID efficiency; cos(#theta); Efficiency", 30, -1, 1)
    h_filteredE_confusionEnergy = ROOT.TH3F("h_filteredE_confusionEnergy", "PID confusion (filtered E); Generated as; Reconstructed as;E (GeV)", 6, -0.5,5.5, 6, -0.5, 5.5,32, 0, 128)
    h_filteredE_confusionCosTheta = ROOT.TH3F("h_filteredE_confusionCosTheta", "PID confusion (filtered E); Generated as; Reconstructed as;cos(#theta)", 6, -0.5,5.5, 6, -0.5, 5.5, 30, -1, 1)
    h_filteredE_efficiencyEnergy = ROOT.TEfficiency("g_filteredE_efficiencyEnergy", "PID efficiency (filtered E); E(GeV); Efficiency", 32, 0, 128)
    h_filteredE_efficiencyCosTheta = ROOT.TEfficiency("g_filteredE_efficiencyCosTheta", "PID efficiency (filtered E); cos(#theta); Efficiency", 30, -1, 1)
    h_cellMC_confusionEnergy = ROOT.TH3F("h_cellMC_confusionEnergy", "PID confusion;Generated as; Reconstructed as;E (GeV)", 6, -0.5,5.5, 6, -0.5, 5.5,32, 0, 128)
    h_cellMC_confusionCosTheta = ROOT.TH3F("h_cellMC_confusionCosTheta", "PID confusion vs cos(#theta); Generated as; Reconstructed as;cos(#theta)", 6, -0.5,5.5, 6, -0.5, 5.5, 30, -1, 1)
    h_cellMC_efficiencyEnergy = ROOT.TEfficiency("g_cellMC_efficiencyEnergy", "PID confusion efficiency; E(GeV); Efficiency", 32, 0, 128)
    h_cellMC_efficiencyCosTheta = ROOT.TEfficiency("g_cellMC_efficiencyCosTheta", "PID efficiency; cos(#theta); Efficiency", 30, -1, 1)
    for axis in h_confusionEnergy.GetXaxis(), h_confusionEnergy.GetYaxis(), h_confusionCosTheta.GetXaxis(), h_confusionCosTheta.GetYaxis()\
        , h_filteredE_confusionEnergy.GetXaxis(), h_filteredE_confusionEnergy.GetYaxis(), h_filteredE_confusionCosTheta.GetXaxis(), h_filteredE_confusionCosTheta.GetYaxis()\
        , h_cellMC_confusionEnergy.GetXaxis(), h_cellMC_confusionEnergy.GetYaxis(), h_cellMC_confusionCosTheta.GetXaxis(), h_cellMC_confusionCosTheta.GetYaxis()\
        , h_efficiencyEnergy.GetYaxis(), h_efficiencyCosTheta.GetYaxis(), h_filteredE_efficiencyEnergy.GetYaxis(), h_filteredE_efficiencyCosTheta.GetYaxis()\
        , h_cellMC_efficiencyEnergy.GetYaxis(), h_cellMC_efficiencyCosTheta.GetYaxis():
        for i, l in map_labels.items():
            axis.SetBinLabel(i+1, l)
            axis.SetLabelSize(0.05)
    ### Fill the hists for all particles that are linked (cluster-based reco<->MC)
    df_confusion = df_reco2mc_links\
        .Define("recoMCpairs_recoPDG","ROOT::VecOps::RVec<int> result; for (auto& p:recoMCpairs) result.push_back(PandoraPFOs[p.first].PDG); return result;")\
        .Define("recoMCpairs_mcPDG","ROOT::VecOps::RVec<int> result; for (auto& p:recoMCpairs) result.push_back(MCParticles[p.second].PDG); return result;")\
        .Define("recoMCpairs_recoE","ROOT::VecOps::RVec<float> result; for (auto& p:recoMCpairs) result.push_back(PandoraPFOs[p.first].energy); return result;")\
        .Define("recoMCpairs_mcE","ROOT::VecOps::RVec<float> result; for (auto& p:recoMCpairs) result.push_back(getEnergy(MCParticles[p.second])); return result;")\
        .Define("recoMCpairs_recoCosTheta","ROOT::VecOps::RVec<float> result; for (auto& p:recoMCpairs) result.push_back(cos(TVector3(PandoraPFOs[p.first].momentum.x,PandoraPFOs[p.first].momentum.y,PandoraPFOs[p.first].momentum.z).Theta())); return result;")
    arrays_confusion = df_confusion.AsNumpy(columns=["recoMCpairs_recoPDG","recoMCpairs_mcPDG", "recoMCpairs_recoE", "recoMCpairs_recoCosTheta", "recoMCpairs_mcE"])
    counter_passed, counter_links, counter_filteredE_passed, counter_filteredE_links = 0,0,0,0
    for iev in range(0,len(arrays_confusion["recoMCpairs_mcPDG"])):
        for ilink in range(0,len(arrays_confusion["recoMCpairs_mcPDG"][iev])):
            if arrays_confusion["recoMCpairs_mcPDG"][iev][ilink] in map_pdgs:
                currentMcPDG = map_pdgs[arrays_confusion["recoMCpairs_mcPDG"][iev][ilink]]
            else:
                currentMcPDG = 5 # others, check if not to extend map_pdgs
            if arrays_confusion["recoMCpairs_recoPDG"][iev][ilink] in map_pdgs:
                currentRecoPDG = map_pdgs[arrays_confusion["recoMCpairs_recoPDG"][iev][ilink]]
            else:
                currentRecoPDG = 5 # others, check if not to extend map_pdgs
            # Fill in histograms with data
            h_confusionEnergy.Fill(currentMcPDG, currentRecoPDG, arrays_confusion["recoMCpairs_recoE"][iev][ilink])
            h_confusionCosTheta.Fill(currentMcPDG, currentRecoPDG, arrays_confusion["recoMCpairs_recoCosTheta"][iev][ilink])
            ifCorrectPID = arrays_confusion["recoMCpairs_mcPDG"][iev][ilink] == arrays_confusion["recoMCpairs_recoPDG"][iev][ilink]
            h_efficiencyEnergy.Fill(ifCorrectPID, arrays_confusion["recoMCpairs_recoE"][iev][ilink])
            h_efficiencyCosTheta.Fill(ifCorrectPID, arrays_confusion["recoMCpairs_recoCosTheta"][iev][ilink])
            if ifCorrectPID:
                counter_passed += 1;
            counter_links += 1;
            # For the "correctly" reco particles (by only energy definition)
            eratio = arrays_confusion["recoMCpairs_recoE"][iev][ilink] / arrays_confusion["recoMCpairs_mcE"][iev][ilink]
            if  eratio > filterEnergy_minThreshold and eratio < filterEnergy_maxThreshold:
                if ifCorrectPID:
                    counter_filteredE_passed += 1;
                counter_filteredE_links += 1;
                h_filteredE_confusionEnergy.Fill(currentMcPDG, currentRecoPDG, arrays_confusion["recoMCpairs_mcE"][iev][ilink])
                h_filteredE_confusionCosTheta.Fill(currentMcPDG, currentRecoPDG, arrays_confusion["recoMCpairs_recoCosTheta"][iev][ilink])
                h_filteredE_efficiencyEnergy.Fill(ifCorrectPID, arrays_confusion["recoMCpairs_mcE"][iev][ilink])
                h_filteredE_efficiencyCosTheta.Fill(ifCorrectPID, arrays_confusion["recoMCpairs_recoCosTheta"][iev][ilink])
    ### Draw different projections of confusion plots
    h_confusionPIDProjection = h_confusionEnergy.Project3D("yx")
    h_confusionCosThetaProjection = h_confusionCosTheta.Project3D("yz")
    h_confusionEProjection = h_confusionEnergy.Project3D("xz")
    h_filteredE_confusionPIDProjection = h_filteredE_confusionEnergy.Project3D("yx")
    h_filteredE_confusionCosThetaProjection = h_filteredE_confusionCosTheta.Project3D("yz")
    h_filteredE_confusionEProjection = h_filteredE_confusionEnergy.Project3D("xz")
    print("Done confusion for all links")

    ## Analyse links at cell level (cell-MC links) instead of taking them from clusters
    # WARNING: hardcoded collections correspond to FcceeCLDo2v07
    ROOT.gInterpreter.Declare("""
    // For all reco particles (rows) and all MC particles (columns) get a ratio of number of cells linked to each MC
    // Calculated as ratio of intersection to union (of corresponding reco and MC particles).
    // To compare with the implementation of MLPF:
    // https://github.com/selvaggi/mlpf/blob/708747daf1c67c8ee8be9ce1dcb4485f5c8c6d55/src/layers/inference_oc.py#L992
    // TODO Check how
    ROOT::VecOps::RVec<std::vector<float>> getMatrixReco2McFromCells(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> reco,
                                             ROOT::VecOps::RVec<edm4hep::ClusterData> clusters,
                                             ROOT::VecOps::RVec<int> cluster_id_cells,
                                             ROOT::VecOps::RVec<edm4hep::CaloHitMCParticleLinkData> links,
                                             ROOT::VecOps::RVec<int> link_id_cell,
                                             ROOT::VecOps::RVec<int> link_id_mc,
                                             ROOT::VecOps::RVec<edm4hep::MCParticleData> mc) {
    // vector (for all reco) of vectors (one per each reco) of number of linked cells based on MC-cell links
    // index in the vector corresponds to MC ID, with the last entry (==mc.size()) corresponding to unlinked (fake) cells
    ROOT::VecOps::RVec<std::vector<float>> matrix_all_reco;

    // First: analyse links to get a map of cell->MC, choosing by the highest weight
    std::map<int,std::vector<std::pair<int,float>>> map_cell2MC_allLinks;
    std::map<int,int> map_cell2MC; // cell_ID <-> mc_ID
    // first fill in the map of all the links to the cell
    for (size_t i=0; i<links.size();i++) map_cell2MC_allLinks[link_id_cell[i]].emplace_back(std::make_pair(link_id_mc[i], links[i].weight));
    // now loop over all links and for cells with more than one MC link, choose the highest weight
    for(auto link: map_cell2MC_allLinks) {
      int id_maxWeight = link.second[0].first;
      double maxWeight = link.second[0].second;
      for(auto v: link.second) {
        if(v.second > maxWeight) {
          maxWeight = v.second;
          id_maxWeight = v.first;
        }
      }
      map_cell2MC[link.first] = id_maxWeight;
    }

    // Get number of MC particles
    auto num_mc = mc.size();

    // Get number of cells per each MC particle for normalisation
    std::vector<int> per_mc_numCells;
    per_mc_numCells.assign(num_mc, 0);
    for(auto link: map_cell2MC) {
         per_mc_numCells[link.second] ++;
    }

    // Get number of cells per each reco particle for normalisation
    std::vector<int> per_reco_numCells;
    per_reco_numCells.assign(reco.size(), 0);
    for (size_t i=0; i<reco.size();i++) {
         for (size_t j = reco[i].clusters_begin; j < reco[i].clusters_end; j++) {
            per_reco_numCells[i] += (clusters[j].hits_end - clusters[j].hits_begin);
         }
    }

    // For each of the reco particles
    for (size_t i=0; i<reco.size();i++) {
         // vector with values for all MC particles
         std::vector<float> per_reco_all_mcs;
         per_reco_all_mcs.assign(num_mc+1, 0); // the last one is for unlinked (fake) cells
         for (size_t j = reco[i].clusters_begin; j < reco[i].clusters_end; j++) {
            for (size_t k = clusters[j].hits_begin; k < clusters[j].hits_end; k++) {
               // look up cluster_id_cells[i] -> MC and then increment per reco vector of the relevant MC particle
               auto linked_MC = num_mc; // If MC is not found
               if(map_cell2MC.contains(cluster_id_cells[i])) {
                 linked_MC = map_cell2MC[cluster_id_cells[i]];
               }
               per_reco_all_mcs[linked_MC] ++;
            }
         }
         // iterate over all MCs to calculate intersection / union ratio:
         for (size_t m=0; m<num_mc;m++) {
            if((per_mc_numCells[m] + per_reco_numCells[i]) > 0)
                per_reco_all_mcs[m] /= (per_mc_numCells[m] + per_reco_numCells[i]);
            else
                per_reco_all_mcs[m] = 0;
         }
         matrix_all_reco.push_back(per_reco_all_mcs);
    }
    return matrix_all_reco;
    }""")
    ROOT.gInterpreter.Declare("""
    // Merge cells from different collections to facilitate further functions
    ROOT::VecOps::RVec<int> mergeCellID(ROOT::VecOps::RVec<podio::ObjectID> cells) {
       // Hardcoded map, TODO ensure it's actually read in from the metadata (f = ROOT.TFile(name)\ f.Get("podio_metadata")\ for i in t:\ names = i.events___idTable.names()\ ids = i.events___idTable.ids()\ And build map from names and ids
       std::map<int,std::string> map_collections = {{1265367388, "ECALBarrel"}, {2257282296, "ECALEndcap"}, {124629188, "HCALBarrel"}, {1286131593, "HCALEndcap"}, {1000433565, "HCALOther" }};
       std::map<int,int> map_collIDtoID =  {{1265367388, 0}, {2257282296, 1}, {124629188, 2}, {1286131593, 3}, {1000433565, 4 }};
       int collSize = map_collIDtoID.size();
       ROOT::VecOps::RVec<int> result;
       for(size_t i = 0; i<cells.size(); i++) {
          result.emplace_back(cells[i].index*collSize+map_collIDtoID[cells[i].collectionID]);
       }
       return result;
    }""")
    map_cell2mc = df.Define("CellIDInClusters","mergeCellID(_PandoraClusters_hits)")\
               .Define("MCIDLinked","ROOT::VecOps::RVec<int> result; for(auto l: _CalohitMCTruthLink_to) {result.emplace_back(l.index);} return result;")\
               .Define("CellIDLinked","mergeCellID(_CalohitMCTruthLink_from)")\
               .Define("cellMCpairs","getMatrixReco2McFromCells(PandoraPFOs, PandoraClusters, CellIDInClusters, CalohitMCTruthLink, CellIDLinked, MCIDLinked, MCParticles)")\
               .Define("cellMCpairs_size","cellMCpairs.size()")\
               .AsNumpy(columns=["cellMCpairs"])["cellMCpairs"]
    map_reco2mc_recoIDs = df_reco2mc_links.AsNumpy(columns=["RecoIDLinked"])["RecoIDLinked"]
    map_reco2mc_mcIDs = df_reco2mc_links.AsNumpy(columns=["MCIDLinked"])["MCIDLinked"]
    list_reco_pdg = df_reco2mc_links.AsNumpy(columns=["recoPDG"])["recoPDG"]
    list_reco_E = df_reco2mc_links.AsNumpy(columns=["recoEnergy"])["recoEnergy"]
    list_mc_pdg = df_reco2mc_links.AsNumpy(columns=["MCPDG"])["MCPDG"]
    list_mc_E = df_reco2mc_links.AsNumpy(columns=["MCEnergy"])["MCEnergy"]
    map_cluster2mc = df_reco2mc_links.AsNumpy(columns=["recoMCpairs"])["recoMCpairs"]
    counter_match_all_cell_cluster_links = 0
    list_mismatch_cell_cluster_links = []
    for i in range( len(list_reco_E)): # N event
        if args.verbose:
            print(f"\033[1m\033[4mEVENT {i}\033[0m")
        if len(list_reco_E[i]) == 0:
            continue
        # investigate per-event matrix with information of number of cells
        # each row represents a reco particle, column an MC particle, with the last column corresponding to a "fake"
        # For each reco-MC pair check ratio of intersection to their union (based on num cells)
        iou_threshold = 0.25 # as in MLPF
        matrix_cell2mc_perEvent = numpy.asarray(map_cell2mc[i])
        matrix_cell2mc_perEvent[matrix_cell2mc_perEvent < iou_threshold] = 0
        row_ind, col_ind = linear_sum_assignment(matrix_cell2mc_perEvent, maximize=True)
        # Borrowed from MLPF: Next three lines remove solutions where there is a shower that is not associated and iou it's zero (or less than threshold)
        mask_matching_matrix = matrix_cell2mc_perEvent[row_ind, col_ind] > 0
        row_ind = row_ind[mask_matching_matrix]
        col_ind = col_ind[mask_matching_matrix]
        # Print info of matched particles
        # A matrix of matched particles from links to cells
        matched_cell = numpy.zeros(numpy.shape(matrix_cell2mc_perEvent))
        matched_cell[row_ind, col_ind] = 1
        # A matrix of matched particles from links to cluster
        matched_cluster = numpy.zeros(numpy.shape(matrix_cell2mc_perEvent))
        for rec in range(len(map_cluster2mc[i])):
            matched_cluster[map_cluster2mc[i][rec].first][map_cluster2mc[i][rec].second] = 1
        # Fill in confusion matrices
    # for event:
    #   for link:
    #      getMappedMC, MappedReco:
    #        if arrays_confusion["recoMCpairs_mcPDG"][iev][ilink] in map_pdgs:
    #            currentMcPDG = map_pdgs[arrays_confusion["recoMCpairs_mcPDG"][iev][ilink]]
    #        else:
    #            currentMcPDG = 5 # others, check if not to extend map_pdgs
    #        if arrays_confusion["recoMCpairs_recoPDG"][iev][ilink] in map_pdgs:
    #            currentRecoPDG = map_pdgs[arrays_confusion["recoMCpairs_recoPDG"][iev][ilink]]
    #        else:
    #            currentRecoPDG = 5 # others, check if not to extend map_pdgs
    #        and fill histograms:
    #        h_confusionEnergy.Fill(currentMcPDG, currentRecoPDG, arrays_confusion["recoMCpairs_recoE"][iev][ilink])
    #        h_confusionCosTheta.Fill(currentMcPDG, currentRecoPDG, arrays_confusion["recoMCpairs_recoCosTheta"][iev][ilink])
    #        ifCorrectPID = arrays_confusion["recoMCpairs_mcPDG"][iev][ilink] == arrays_confusion["recoMCpairs_recoPDG"][iev][ilink]
    #        h_efficiencyEnergy.Fill(ifCorrectPID, arrays_confusion["recoMCpairs_recoE"][iev][ilink])
    #        h_efficiencyCosTheta.Fill(ifCorrectPID, arrays_confusion["recoMCpairs_recoCosTheta"][iev][ilink])
        # Compare individual cluster-level links to cell-level links
        if (matched_cell == matched_cluster).all():
            if args.verbose:
                print("\033[38;5;28mAll linkes match!\033[0m")
            counter_match_all_cell_cluster_links += 1
        else:
            list_mismatch_cell_cluster_links.append(i)
            if args.verbose:
                print("\033[38;5;1mCell/cluster link discrepancy!\033[0m")
                array_to_print=[]
                header_to_print = [""]
                mc_threshold = 0.1 #GeV
                below_threshold = 0
                for r in range(numpy.shape(matched_cell)[0]):
                    row_to_print = [f"{list_reco_pdg[i][int(map_cluster2mc[i][r].first)]}, {list_reco_E[i][int(map_cluster2mc[i][r].first)]:.1f} GeV"]
                    for c in range(numpy.shape(matched_cell)[1] - 1): # -1 to ignore the "fakes"
                        if list_mc_E[i][c] > mc_threshold:
                            if matched_cell[r][c] == matched_cluster[r][c]:
                                row_to_print.append( f" \033[38;5;28m{matched_cluster[r][c]:.0f}\033[0m ")
                            else:
                                row_to_print.append( f" \033[38;5;1m{matched_cell[r][c]:.0f}\033[0m(cell) \033[38;5;1m{matched_cluster[r][c]:.0f}\033[0m(cluster) ")
                    row_to_print.append("") # to display the MCs below threshold"
                    array_to_print.append(row_to_print)
                for mc in range(len(list_mc_E[i])):
                    if list_mc_E[i][mc] > mc_threshold:
                        header_to_print.append(f'{list_mc_pdg[i][mc]}, {list_mc_E[i][mc]:.1f} GeV')
                    else:
                        below_threshold += 1
                header_to_print.append(f"and {below_threshold} MCs < {mc_threshold:.1f} GeV")
                print(tabulate(array_to_print, headers=header_to_print))

    print(f"Ratio of events with identical links between clusters and cells: {counter_match_all_cell_cluster_links/len(list_reco_E)}")


    # For total efficiency
    if counter_links:
        total_efficiency = counter_passed / counter_links
        filtered_efficiency = counter_filteredE_passed / counter_filteredE_links
        print(f"Total efficiency (counted from all the reco<->MC links): {total_efficiency}")
        print(f"Filtered energy efficiency (counted from the reco<->MC links that recunstruct energy within {recoResult_resolution/recoResult_mean*100:.1f}%): {filtered_efficiency}")

    # Store
    # outfile = ROOT.TFile(outname, "RECREATE")
    # outfile.cd()
    # h_simE.Write("energy_sim")
    # h_recoHighestE.Write("energy_reco")

    # h_simEnergyRatio.Write("energy_sim_ratio")
    # store_sim_mean = numpy.zeros(1, dtype=float)
    # store_sim_meanErr = numpy.zeros(1, dtype=float)
    # store_sim_resolution = numpy.zeros(1, dtype=float)
    # store_sim_resolutionErr = numpy.zeros(1, dtype=float)
    # store_reco_mean = numpy.zeros(1, dtype=float)
    # store_reco_meanErr = numpy.zeros(1, dtype=float)
    # store_reco_resolution = numpy.zeros(1, dtype=float)
    # store_reco_resolutionErr = numpy.zeros(1, dtype=float)
    # store_EMC = numpy.zeros(1, dtype=float)
    # store_efficiency = numpy.zeros(1, dtype=float)
    # tree = ROOT.TTree("results", "Fit parameters foor single pgun analysis")
    # tree.Branch("sim_en_mean", store_sim_mean, "store_sim_mean/D");
    # tree.Branch("sim_en_meanErr", store_sim_meanErr, "store_sim_meanErr/D");
    # tree.Branch("sim_en_resolution", store_sim_resolution, "store_sim_resolution/D");
    # tree.Branch("sim_en_resolutionErr", store_sim_resolutionErr, "store_sim_resolutionErr/D");
    # tree.Branch("reco_en_mean", store_reco_mean, "store_reco_mean/D");
    # tree.Branch("reco_en_meanErr", store_reco_meanErr, "store_reco_meanErr/D");
    # tree.Branch("reco_en_resolution", store_reco_resolution, "store_reco_resolution/D");
    # tree.Branch("reco_en_resolutionErr", store_reco_resolutionErr, "store_reco_resolutionErr/D");
    # tree.Branch("enMC", store_EMC, "store_EMC/D");
    # tree.Branch("efficiency", store_efficiency, "store_efficiency/D");
    # store_sim_mean[0] = simResult_mean
    # store_sim_meanErr[0] = simResult_meanError
    # store_sim_resolution[0] = simResult_resolution
    # store_sim_resolutionErr[0] = simResult_resolutionError
    # store_reco_mean[0] = recoResult_mean
    # store_reco_meanErr[0] = recoResult_meanError
    # store_reco_resolution[0] = recoResult_resolution
    # store_reco_resolutionErr[0] = recoResult_resolutionError
    # store_EMC[0] = gun_mean
    # store_efficiency[0] = total_efficiency
    # tree.Fill()
    # tree.Write()
    # outfile.Close()

    ## Draw
    # once we use root 6.34colours = [ROOT.kP6Blue, ROOT.kP6Yellow, ROOT.kP6Red, ROOT.kP6Grape, ROOT.kP6Gray]
    colours = [ROOT.kAzure+1, ROOT.kOrange-2, ROOT.kRed-4, ROOT.kMagenta-2, ROOT.kGray+1]
    canv = ROOT.TCanvas("canv","Basic reconstruction control plots for MC<->reco links",1200,900)
    canv.Divide(2,2)
    canv.cd(1) # draw energy distributions
    h_recoRatioE.SetLineColor(colours[0])
    h_recoRatioE.SetFillColor(colours[0])
    f_recoFit.SetLineColor(colours[0])
    h_recoRatioE.Draw()
    f_recoFit.Draw("same")
    h_recoRatioE.SetTitle("Energy ratio")
    h_recoRatioE.GetXaxis().SetTitle("E_{PFO} / E_{MC}")
    legend1 = ROOT.TLegend(0.1,0.8,0.9,0.9)
    legend1.AddEntry(f_recoFit.GetName(),f"fit defining the filter: #mu={recoResult_mean:.1f}, #sigma={recoResult_resolution:.3f}","l")
    legend1.Draw()
    canv.cd(2)
    h_diffE.Draw()
    h_diffE.SetTitle("Energy difference")
    h_diffE.GetXaxis().SetTitle("E_{PFO} - E_{MC} (GeV)")
    canv.cd(3)
    h_diffTheta.Draw()
    h_diffTheta.SetTitle("#theta difference")
    h_diffTheta.GetXaxis().SetTitle("#theta_{PFO} - #theta_{MC} (rad)")
    canv.cd(4)
    h_diffPhi.Draw()
    h_diffPhi.SetTitle("#phi difference")
    h_diffPhi.GetXaxis().SetTitle("#phi_{PFO} - #phi_{MC} (rad)")


    canvConf = ROOT.TCanvas("canvConf","Reconstruction control plots for MC<->reco links",2100,900)
    canvConf.Divide(4,2)
    canvConf.cd(1)
    h_confusionPIDProjection.Draw("colztext")
    canvConf.cd(2)
    h_confusionCosThetaProjection.Draw("colztext")
    canvConf.cd(3)
    h_confusionEProjection.Draw("colztext")
    canvConf.cd(5)
    h_filteredE_confusionPIDProjection.Draw("colztext")
    canvConf.cd(6)
    h_filteredE_confusionCosThetaProjection.Draw("colztext")
    canvConf.cd(7)
    h_filteredE_confusionEProjection.Draw("colztext")
    pad10 = canvConf.cd(4) # Efficiency vs cosTheta
    h_efficiencyCosTheta.SetLineColor(colours[1])
    h_filteredE_efficiencyCosTheta.SetLineColor(colours[2])
    h_efficiencyCosTheta.SetMarkerColor(colours[1])
    h_filteredE_efficiencyCosTheta.SetMarkerColor(colours[2])
    h_efficiencyCosTheta.SetMarkerStyle(24)
    h_filteredE_efficiencyCosTheta.SetMarkerStyle(22)
    h_efficiencyCosTheta.Draw("aep")
    h_filteredE_efficiencyCosTheta.Draw("sameep")
    legendEffT = ROOT.TLegend(0.3,0.2,0.9,0.4)
    legendEffT.AddEntry(h_efficiencyCosTheta.GetName(),"all links")
    legendEffT.AddEntry(h_filteredE_efficiencyCosTheta.GetName(),"filtered energy")
    legendEffT.Draw()
    pad11 = canvConf.cd(8) # Efficiency vs E
    h_efficiencyEnergy.SetLineColor(colours[1])
    h_filteredE_efficiencyEnergy.SetLineColor(colours[2])
    h_efficiencyEnergy.SetMarkerColor(colours[1])
    h_filteredE_efficiencyEnergy.SetMarkerColor(colours[2])
    h_efficiencyEnergy.SetMarkerStyle(24)
    h_filteredE_efficiencyEnergy.SetMarkerStyle(22)
    h_efficiencyEnergy.Draw("aep")
    h_filteredE_efficiencyEnergy.Draw("sameep")
    legendEffE = ROOT.TLegend(0.3,0.2,0.9,0.4)
    legendEffE.AddEntry(h_efficiencyEnergy.GetName(),"all links")
    legendEffE.AddEntry(h_filteredE_efficiencyEnergy.GetName(),"filtered energy")
    legendEffE.Draw()
    canv.Update()
    canv.SaveAs(f"recoValidation_histograms_{inputlist[0].split('/')[-1:][0][:-5]}.pdf")

    # Draw and store multiplicity control plots
    canvMultiplicity = ROOT.TCanvas("canvMultiplicity","Reconstruction control plots for multiplicity of MC/reco particles",1200,900)
    canvMultiplicity.Divide(2,2)
    canvMultiplicity.cd(1)
    h_numClusters.Draw()
    canvMultiplicity.cd(2)
    h_numCells.Draw()
    padNum = canvMultiplicity.cd(3)
    h_numMC.SetLineColor(colours[0])
    h_numMC.SetFillColor(colours[0])
    h_numReco.SetLineColor(colours[1])
    h_numReco.SetLineWidth(2)
    padNum.SetLogy()
    h_numMC.Draw()
    h_numReco.Draw("same")
    h_numClusterLinks.Draw("same")
    legend3 = ROOT.TLegend(0.3,0.7,0.9,0.9)
    legend3.AddEntry(h_numMC.GetName(),"MC > 100 MeV")
    legend3.AddEntry(h_numReco.GetName(),"reco > 100 MeV")
    legend3.Draw()
    padNum2 = canvMultiplicity.cd(4)
    padNum2.SetLogy()
    h_numClusterLinks.SetLineColor(colours[0])
    h_numClusterLinks.SetLineWidth(2)
    h_numClusterLinks_filtered.SetLineColor(colours[1])
    h_numClusterLinks_filtered.SetLineWidth(2)
    h_numClusterLinks.Draw()
    h_numClusterLinks_filtered.Draw("same")
    legend4 = ROOT.TLegend(0.3,0.7,0.9,0.9)
    legend4.AddEntry(h_numClusterLinks.GetName(),f"all cluster links reco<->MC")
    legend4.AddEntry(h_numClusterLinks_filtered.GetName(),f"cluster links E_{{PFO}} #in ({recoResult_mean:.1f}#pm{recoResult_resolution/recoResult_mean*100:.1f}%) E_{{MC}} ")
    legend4.Draw()
    if args.interactive:
        input("")

if __name__ == "__main__":

    run(args.infile, args.outfile, args.ncpus)
