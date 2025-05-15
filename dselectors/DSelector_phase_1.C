#include "DSelector_phase_1.h"

void DSelector_phase_1::Init(TTree *locTree) {

    dFlatTreeFileName = "flat.root";
    dFlatTreeName = "kin";
    dSaveDefaultFlatBranches = false;
    bool locInitializedPriorFlag = dInitializedFlag;
    DSelector::Init(locTree);
    if(locInitializedPriorFlag) {{
        return;
    }}
    Get_ComboWrappers();
    dPreviousRunNumber = 0;
    dAnalysisActions.push_back(new DHistogramAction_ParticleID(dComboWrapper, false));
    dAnalysisActions.push_back(new DHistogramAction_PIDFOM(dComboWrapper));
    dAnalysisActions.push_back(new DHistogramAction_KinFitResults(dComboWrapper));
    dAnalysisActions.push_back(new DHistogramAction_BeamEnergy(dComboWrapper, false));
    dAnalysisActions.push_back(new DHistogramAction_ParticleComboKinematics(dComboWrapper, false));

    Initialize_Actions();

    dFlatTreeInterface->Create_Branch_Fundamental<UInt_t>("RunNumber");
    dFlatTreeInterface->Create_Branch_Fundamental<ULong64_t>("EventNumber");
    dFlatTreeInterface->Create_Branch_Fundamental<UInt_t>("ComboNumber");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Weight");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_0_E");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_0_Px");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_0_Py");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_0_Pz");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_1_E");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_1_Px");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_1_Py");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_1_Pz");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_2_E");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_2_Px");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_2_Py");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_2_Pz");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_3_E");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_3_Px");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_3_Py");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_3_Pz");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_4_E");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_4_Px");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_4_Py");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_4_Pz");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_5_E");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_5_Px");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_5_Py");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_5_Pz");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_6_E");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_6_Px");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_6_Py");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_6_Pz");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_7_E");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_7_Px");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_7_Py");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("P4_7_Pz");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("ME");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("MM2");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("RFL1");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("RFL2");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("ChiSqDOF");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("RF");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_Z");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_P");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_Theta");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_dEdx_CDC");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_dEdx_CDC_integral");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_dEdx_FDC");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_dEdx_ST");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_dEdx_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_E_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_E_FCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_DeltaT_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_DeltaT_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_DeltaT_FCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_Beta_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_Beta_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("Proton_Beta_FCAL");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_P");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_Theta");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_dEdx_CDC");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_dEdx_CDC_integral");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_dEdx_FDC");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_dEdx_ST");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_dEdx_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_E_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_E_FCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_DeltaT_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_DeltaT_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_DeltaT_FCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_Beta_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_Beta_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus1_Beta_FCAL");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_P");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_Theta");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_dEdx_CDC");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_dEdx_CDC_integral");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_dEdx_FDC");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_dEdx_ST");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_dEdx_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_E_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_E_FCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_DeltaT_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_DeltaT_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_DeltaT_FCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_Beta_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_Beta_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus1_Beta_FCAL");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_P");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_Theta");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_dEdx_CDC");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_dEdx_CDC_integral");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_dEdx_FDC");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_dEdx_ST");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_dEdx_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_E_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_E_FCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_DeltaT_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_DeltaT_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_DeltaT_FCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_Beta_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_Beta_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiPlus2_Beta_FCAL");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_P");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_Theta");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_dEdx_CDC");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_dEdx_CDC_integral");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_dEdx_FDC");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_dEdx_ST");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_dEdx_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_E_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_E_FCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_DeltaT_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_DeltaT_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_DeltaT_FCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_Beta_BCAL");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_Beta_TOF");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("PiMinus2_Beta_FCAL");

    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("KShort1_Z");
    dFlatTreeInterface->Create_Branch_Fundamental<Float_t>("KShort2_Z");
}

Bool_t DSelector_phase_1::Process(Long64_t locEntry) {

    DSelector::Process(locEntry);

    // If the run number changes, use RCDB to get polarization info:
    UInt_t locRunNumber = Get_RunNumber();
    if(locRunNumber != dPreviousRunNumber) {
        dIsPolarizedFlag = dAnalysisUtilities.Get_IsPolarizedBeam(locRunNumber, dIsPARAFlag);
        dPreviousRunNumber = locRunNumber;
    }


    Reset_Actions_NewEvent();

    for(UInt_t loc_i = 0; loc_i < Get_NumCombos(); ++loc_i) {

        dComboWrapper->Set_ComboIndex(loc_i);
        if(dComboWrapper->Get_IsComboCut()) {
            continue; // Ignore all combos cut in any previous analysis
        }

        // Step 0
        Int_t locBeamID = dComboBeamWrapper->Get_BeamID();
        Int_t locProtonTrackID = dProtonWrapper->Get_TrackID();

        // Step 1
        Int_t locPiMinus1TrackID = dPiMinus1Wrapper->Get_TrackID();
        Int_t locPiPlus1TrackID = dPiPlus1Wrapper->Get_TrackID();

        // Step 2
        Int_t locPiMinus2TrackID = dPiMinus2Wrapper->Get_TrackID();
        Int_t locPiPlus2TrackID = dPiPlus2Wrapper->Get_TrackID();

        // Step 0
        TLorentzVector locBeamP4 = dComboBeamWrapper->Get_P4();
        TLorentzVector locBeamP4_Measured = dComboBeamWrapper->Get_P4_Measured();
        TLorentzVector locBeamX4_Measured = dComboBeamWrapper->Get_X4_Measured();
        TLorentzVector locProtonP4 = dProtonWrapper->Get_P4();
        TLorentzVector locProtonP4_Measured = dProtonWrapper->Get_P4_Measured();
        TLorentzVector locProtonX4_Measured = dProtonWrapper->Get_X4_Measured();

        // Step 1
        TLorentzVector locDecayingKShort1P4 = dDecayingKShort1Wrapper->Get_P4();
        TLorentzVector locDecayingKShort1X4 = dDecayingKShort1Wrapper->Get_X4();
        TLorentzVector locPiMinus1P4 = dPiMinus1Wrapper->Get_P4();
        TLorentzVector locPiMinus1P4_Measured = dPiMinus1Wrapper->Get_P4_Measured();
        TLorentzVector locPiMinus1X4_Measured = dPiMinus1Wrapper->Get_X4_Measured();
        TLorentzVector locPiPlus1P4 = dPiPlus1Wrapper->Get_P4();
        TLorentzVector locPiPlus1P4_Measured = dPiPlus1Wrapper->Get_P4_Measured();
        TLorentzVector locPiPlus1X4_Measured = dPiPlus1Wrapper->Get_X4_Measured();

        // Step 2
        TLorentzVector locDecayingKShort2P4 = dDecayingKShort2Wrapper->Get_P4();
        TLorentzVector locDecayingKShort2X4 = dDecayingKShort2Wrapper->Get_X4();
        TLorentzVector locPiMinus2P4 = dPiMinus2Wrapper->Get_P4();
        TLorentzVector locPiMinus2P4_Measured = dPiMinus2Wrapper->Get_P4_Measured();
        TLorentzVector locPiMinus2X4_Measured = dPiMinus2Wrapper->Get_X4_Measured();
        TLorentzVector locPiPlus2P4 = dPiPlus2Wrapper->Get_P4();
        TLorentzVector locPiPlus2P4_Measured = dPiPlus2Wrapper->Get_P4_Measured();
        TLorentzVector locPiPlus2X4_Measured = dPiPlus2Wrapper->Get_X4_Measured();


        // Boost:  -> COM
        TLorentzVector locBoostP4_COM = locBeamP4 + dTargetP4;
        TLorentzVector locBeamP4_COM = locBeamP4;
        TLorentzVector locProtonP4_COM = locProtonP4;
        TLorentzVector locDecayingKShort1P4_COM = locDecayingKShort1P4;
        TLorentzVector locPiMinus1P4_COM = locPiMinus1P4;
        TLorentzVector locPiPlus1P4_COM = locPiPlus1P4;
        TLorentzVector locDecayingKShort2P4_COM = locDecayingKShort2P4;
        TLorentzVector locPiMinus2P4_COM = locPiMinus2P4;
        TLorentzVector locPiPlus2P4_COM = locPiPlus2P4;

        locBeamP4_COM.Boost(-locBoostP4_COM.BoostVector());
        locProtonP4_COM.Boost(-locBoostP4_COM.BoostVector());
        locDecayingKShort1P4_COM.Boost(-locBoostP4_COM.BoostVector());
        locPiMinus1P4_COM.Boost(-locBoostP4_COM.BoostVector());
        locPiPlus1P4_COM.Boost(-locBoostP4_COM.BoostVector());
        locDecayingKShort2P4_COM.Boost(-locBoostP4_COM.BoostVector());
        locPiPlus2P4_COM.Boost(-locBoostP4_COM.BoostVector());


        // Accidentals:
        Double_t locDeltaT_RF = dAnalysisUtilities.Get_DeltaT_RF(Get_RunNumber(), locBeamX4_Measured, dComboWrapper);

        // ChiSqDOF
        Double_t locChiSqDOF;
        locChiSqDOF = dComboWrapper->Get_ChiSq_KinFit("") / dComboWrapper->Get_NDF_KinFit("");

        // MM2/ME
        TLorentzVector locMissingP4 = locBeamP4_Measured + dTargetP4 - locProtonP4_Measured - locPiMinus1P4_Measured - locPiPlus1P4_Measured - locPiMinus2P4_Measured - locPiPlus2P4_Measured;

        // RFL
        TLorentzVector locPrimarySpacetimeVertex1, locPrimarySpacetimeVertex2, locSecondarySpacetimeVertex1, locSecondarySpacetimeVertex2;
        int locPrimaryIndex1 = dStep1Wrapper->Get_InitDecayFromIndices().first;
        int locPrimaryIndex2 = dStep2Wrapper->Get_InitDecayFromIndices().first;
        locSecondarySpacetimeVertex1 = dStep1Wrapper->Get_X4();
        locSecondarySpacetimeVertex2 = dStep2Wrapper->Get_X4();
        locPrimarySpacetimeVertex1 = dComboWrapper->Get_ParticleComboStep(locPrimaryIndex1)->Get_X4();
        locPrimarySpacetimeVertex2 = dComboWrapper->Get_ParticleComboStep(locPrimaryIndex2)->Get_X4();
        TLorentzVector locKShort1_DX4 = locSecondarySpacetimeVertex1 - locPrimarySpacetimeVertex1;
        TLorentzVector locKShort2_DX4 = locSecondarySpacetimeVertex2 - locPrimarySpacetimeVertex2;
        Double_t locKShort1PL = locKShort1_DX4.Vect().Mag();
        Double_t locKShort2PL = locKShort2_DX4.Vect().Mag();
        Double_t locKShort1RFL = locKShort1PL * ParticleMass(KShort) / (29.9792458 * locDecayingKShort1P4.P());
        Double_t locKShort2RFL = locKShort2PL * ParticleMass(KShort) / (29.9792458 * locDecayingKShort2P4.P());

        // Detectors_Proton
        Double_t locProton_P = locProtonP4_Measured.P();
        Double_t locProton_Theta = locProtonP4_Measured.Theta();
        Double_t locProton_dEdx_CDC = dProtonWrapper->Get_dEdx_CDC() * 1e6;
        Double_t locProton_dEdx_CDC_integral = dProtonWrapper->Get_dEdx_CDC_integral() * 1e6;
        Double_t locProton_dEdx_FDC = dProtonWrapper->Get_dEdx_FDC() * 1e6;
        Double_t locProton_dEdx_ST = dProtonWrapper->Get_dEdx_ST() * 1e3;
        Double_t locProton_dEdx_TOF = dProtonWrapper->Get_dEdx_TOF() * 1e3;
        Double_t locProton_E_BCAL = dProtonWrapper->Get_Energy_BCAL();
        Double_t locProton_E_FCAL = dProtonWrapper->Get_Energy_FCAL();
        
        Double_t locProton_DeltaT_BCAL = 0.0;
        Double_t locProton_DeltaT_TOF = 0.0;
        Double_t locProton_DeltaT_FCAL = 0.0;
        Double_t locProton_Beta_BCAL = 0.0;
        Double_t locProton_Beta_TOF = 0.0;
        Double_t locProton_Beta_FCAL = 0.0;
        
        Double_t locProton_DeltaT = locProtonX4_Measured.T() - dComboWrapper->Get_RFTime_Measured() - (locProtonX4_Measured.Z() - dComboWrapper->Get_TargetCenter().Z())/29.9792458;
        Double_t locProton_Beta = dProtonWrapper->Get_Beta_Timing_Measured();
        
        if (dProtonWrapper->Get_Detector_System_Timing() == SYS_BCAL) {
            locProton_DeltaT_BCAL = locProton_DeltaT;
            locProton_Beta_BCAL = locProton_Beta;
        } else if (dProtonWrapper->Get_Detector_System_Timing() == SYS_TOF) {
            locProton_DeltaT_TOF = locProton_DeltaT;
            locProton_Beta_TOF = locProton_Beta;
        } else if (dProtonWrapper->Get_Detector_System_Timing() == SYS_FCAL) {
            locProton_DeltaT_FCAL = locProton_DeltaT;
            locProton_Beta_FCAL = locProton_Beta;
        }

        // Detectors_PiPlus1
        Double_t locPiPlus1_P = locPiPlus1P4_Measured.P();
        Double_t locPiPlus1_Theta = locPiPlus1P4_Measured.Theta();
        Double_t locPiPlus1_dEdx_CDC = dPiPlus1Wrapper->Get_dEdx_CDC() * 1e6;
        Double_t locPiPlus1_dEdx_CDC_integral = dPiPlus1Wrapper->Get_dEdx_CDC_integral() * 1e6;
        Double_t locPiPlus1_dEdx_FDC = dPiPlus1Wrapper->Get_dEdx_FDC() * 1e6;
        Double_t locPiPlus1_dEdx_ST = dPiPlus1Wrapper->Get_dEdx_ST() * 1e3;
        Double_t locPiPlus1_dEdx_TOF = dPiPlus1Wrapper->Get_dEdx_TOF() * 1e3;
        Double_t locPiPlus1_E_BCAL = dPiPlus1Wrapper->Get_Energy_BCAL();
        Double_t locPiPlus1_E_FCAL = dPiPlus1Wrapper->Get_Energy_FCAL();
        
        Double_t locPiPlus1_DeltaT_BCAL = 0.0;
        Double_t locPiPlus1_DeltaT_TOF = 0.0;
        Double_t locPiPlus1_DeltaT_FCAL = 0.0;
        Double_t locPiPlus1_Beta_BCAL = 0.0;
        Double_t locPiPlus1_Beta_TOF = 0.0;
        Double_t locPiPlus1_Beta_FCAL = 0.0;
        
        Double_t locPiPlus1_DeltaT = locPiPlus1X4_Measured.T() - dComboWrapper->Get_RFTime_Measured() - (locPiPlus1X4_Measured.Z() - dComboWrapper->Get_TargetCenter().Z())/29.9792458;
        Double_t locPiPlus1_Beta = dPiPlus1Wrapper->Get_Beta_Timing_Measured();
        
        if (dPiPlus1Wrapper->Get_Detector_System_Timing() == SYS_BCAL) {
            locPiPlus1_DeltaT_BCAL = locPiPlus1_DeltaT;
            locPiPlus1_Beta_BCAL = locPiPlus1_Beta;
        } else if (dPiPlus1Wrapper->Get_Detector_System_Timing() == SYS_TOF) {
            locPiPlus1_DeltaT_TOF = locPiPlus1_DeltaT;
            locPiPlus1_Beta_TOF = locPiPlus1_Beta;
        } else if (dPiPlus1Wrapper->Get_Detector_System_Timing() == SYS_FCAL) {
            locPiPlus1_DeltaT_FCAL = locPiPlus1_DeltaT;
            locPiPlus1_Beta_FCAL = locPiPlus1_Beta;
        }

        // Detectors_PiMinus1
        Double_t locPiMinus1_P = locPiMinus1P4_Measured.P();
        Double_t locPiMinus1_Theta = locPiMinus1P4_Measured.Theta();
        Double_t locPiMinus1_dEdx_CDC = dPiMinus1Wrapper->Get_dEdx_CDC() * 1e6;
        Double_t locPiMinus1_dEdx_CDC_integral = dPiMinus1Wrapper->Get_dEdx_CDC_integral() * 1e6;
        Double_t locPiMinus1_dEdx_FDC = dPiMinus1Wrapper->Get_dEdx_FDC() * 1e6;
        Double_t locPiMinus1_dEdx_ST = dPiMinus1Wrapper->Get_dEdx_ST() * 1e3;
        Double_t locPiMinus1_dEdx_TOF = dPiMinus1Wrapper->Get_dEdx_TOF() * 1e3;
        Double_t locPiMinus1_E_BCAL = dPiMinus1Wrapper->Get_Energy_BCAL();
        Double_t locPiMinus1_E_FCAL = dPiMinus1Wrapper->Get_Energy_FCAL();
        
        Double_t locPiMinus1_DeltaT_BCAL = 0.0;
        Double_t locPiMinus1_DeltaT_TOF = 0.0;
        Double_t locPiMinus1_DeltaT_FCAL = 0.0;
        Double_t locPiMinus1_Beta_BCAL = 0.0;
        Double_t locPiMinus1_Beta_TOF = 0.0;
        Double_t locPiMinus1_Beta_FCAL = 0.0;
        
        Double_t locPiMinus1_DeltaT = locPiMinus1X4_Measured.T() - dComboWrapper->Get_RFTime_Measured() - (locPiMinus1X4_Measured.Z() - dComboWrapper->Get_TargetCenter().Z())/29.9792458;
        Double_t locPiMinus1_Beta = dPiMinus1Wrapper->Get_Beta_Timing_Measured();
        
        if (dPiMinus1Wrapper->Get_Detector_System_Timing() == SYS_BCAL) {
            locPiMinus1_DeltaT_BCAL = locPiMinus1_DeltaT;
            locPiMinus1_Beta_BCAL = locPiMinus1_Beta;
        } else if (dPiMinus1Wrapper->Get_Detector_System_Timing() == SYS_TOF) {
            locPiMinus1_DeltaT_TOF = locPiMinus1_DeltaT;
            locPiMinus1_Beta_TOF = locPiMinus1_Beta;
        } else if (dPiMinus1Wrapper->Get_Detector_System_Timing() == SYS_FCAL) {
            locPiMinus1_DeltaT_FCAL = locPiMinus1_DeltaT;
            locPiMinus1_Beta_FCAL = locPiMinus1_Beta;
        }

        // Detectors_PiPlus2
        Double_t locPiPlus2_P = locPiPlus2P4_Measured.P();
        Double_t locPiPlus2_Theta = locPiPlus2P4_Measured.Theta();
        Double_t locPiPlus2_dEdx_CDC = dPiPlus2Wrapper->Get_dEdx_CDC() * 1e6;
        Double_t locPiPlus2_dEdx_CDC_integral = dPiPlus2Wrapper->Get_dEdx_CDC_integral() * 1e6;
        Double_t locPiPlus2_dEdx_FDC = dPiPlus2Wrapper->Get_dEdx_FDC() * 1e6;
        Double_t locPiPlus2_dEdx_ST = dPiPlus2Wrapper->Get_dEdx_ST() * 1e3;
        Double_t locPiPlus2_dEdx_TOF = dPiPlus2Wrapper->Get_dEdx_TOF() * 1e3;
        Double_t locPiPlus2_E_BCAL = dPiPlus2Wrapper->Get_Energy_BCAL();
        Double_t locPiPlus2_E_FCAL = dPiPlus2Wrapper->Get_Energy_FCAL();
        
        Double_t locPiPlus2_DeltaT_BCAL = 0.0;
        Double_t locPiPlus2_DeltaT_TOF = 0.0;
        Double_t locPiPlus2_DeltaT_FCAL = 0.0;
        Double_t locPiPlus2_Beta_BCAL = 0.0;
        Double_t locPiPlus2_Beta_TOF = 0.0;
        Double_t locPiPlus2_Beta_FCAL = 0.0;
        
        Double_t locPiPlus2_DeltaT = locPiPlus2X4_Measured.T() - dComboWrapper->Get_RFTime_Measured() - (locPiPlus2X4_Measured.Z() - dComboWrapper->Get_TargetCenter().Z())/29.9792458;
        Double_t locPiPlus2_Beta = dPiPlus2Wrapper->Get_Beta_Timing_Measured();
        
        if (dPiPlus2Wrapper->Get_Detector_System_Timing() == SYS_BCAL) {
            locPiPlus2_DeltaT_BCAL = locPiPlus2_DeltaT;
            locPiPlus2_Beta_BCAL = locPiPlus2_Beta;
        } else if (dPiPlus2Wrapper->Get_Detector_System_Timing() == SYS_TOF) {
            locPiPlus2_DeltaT_TOF = locPiPlus2_DeltaT;
            locPiPlus2_Beta_TOF = locPiPlus2_Beta;
        } else if (dPiPlus2Wrapper->Get_Detector_System_Timing() == SYS_FCAL) {
            locPiPlus2_DeltaT_FCAL = locPiPlus2_DeltaT;
            locPiPlus2_Beta_FCAL = locPiPlus2_Beta;
        }

        // Detectors_PiMinus2
        Double_t locPiMinus2_P = locPiMinus2P4_Measured.P();
        Double_t locPiMinus2_Theta = locPiMinus2P4_Measured.Theta();
        Double_t locPiMinus2_dEdx_CDC = dPiMinus2Wrapper->Get_dEdx_CDC() * 1e6;
        Double_t locPiMinus2_dEdx_CDC_integral = dPiMinus2Wrapper->Get_dEdx_CDC_integral() * 1e6;
        Double_t locPiMinus2_dEdx_FDC = dPiMinus2Wrapper->Get_dEdx_FDC() * 1e6;
        Double_t locPiMinus2_dEdx_ST = dPiMinus2Wrapper->Get_dEdx_ST() * 1e3;
        Double_t locPiMinus2_dEdx_TOF = dPiMinus2Wrapper->Get_dEdx_TOF() * 1e3;
        Double_t locPiMinus2_E_BCAL = dPiMinus2Wrapper->Get_Energy_BCAL();
        Double_t locPiMinus2_E_FCAL = dPiMinus2Wrapper->Get_Energy_FCAL();
        
        Double_t locPiMinus2_DeltaT_BCAL = 0.0;
        Double_t locPiMinus2_DeltaT_TOF = 0.0;
        Double_t locPiMinus2_DeltaT_FCAL = 0.0;
        Double_t locPiMinus2_Beta_BCAL = 0.0;
        Double_t locPiMinus2_Beta_TOF = 0.0;
        Double_t locPiMinus2_Beta_FCAL = 0.0;
        
        Double_t locPiMinus2_DeltaT = locPiMinus2X4_Measured.T() - dComboWrapper->Get_RFTime_Measured() - (locPiMinus2X4_Measured.Z() - dComboWrapper->Get_TargetCenter().Z())/29.9792458;
        Double_t locPiMinus2_Beta = dPiMinus2Wrapper->Get_Beta_Timing_Measured();
        
        if (dPiMinus2Wrapper->Get_Detector_System_Timing() == SYS_BCAL) {
            locPiMinus2_DeltaT_BCAL = locPiMinus2_DeltaT;
            locPiMinus2_Beta_BCAL = locPiMinus2_Beta;
        } else if (dPiMinus2Wrapper->Get_Detector_System_Timing() == SYS_TOF) {
            locPiMinus2_DeltaT_TOF = locPiMinus2_DeltaT;
            locPiMinus2_Beta_TOF = locPiMinus2_Beta;
        } else if (dPiMinus2Wrapper->Get_Detector_System_Timing() == SYS_FCAL) {
            locPiMinus2_DeltaT_FCAL = locPiMinus2_DeltaT;
            locPiMinus2_Beta_FCAL = locPiMinus2_Beta;
        }

        if(!Execute_Actions()) {
            continue;
        }

        // CUT: select coherent peak
        if(locBeamP4.E() < 8.2 || locBeamP4.E() > 8.8) {
            dComboWrapper->Set_IsComboCut(true);
        }

        // CUT: remove unpolarized events (AMO)
        if(!dIsPolarizedFlag) {
            dComboWrapper->Set_IsComboCut(true);
        }

        // Don't output combos that were cut
        if(dComboWrapper->Get_IsComboCut()) {
            continue;
        }

        // Fill labeling branches
        dFlatTreeInterface->Fill_Fundamental<UInt_t>("RunNumber", locRunNumber);
        dFlatTreeInterface->Fill_Fundamental<ULong64_t>("EventNumber", Get_EventNumber());
        dFlatTreeInterface->Fill_Fundamental<UInt_t>("ComboNumber", loc_i);

        // Fill Flat Weight branch
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Weight", 1.0);

        // Fill Flat P4_0 branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_0_E", locBeamP4_COM.E());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_0_Px", locBeamP4_COM.Px());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_0_Py", locBeamP4_COM.Py());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_0_Pz", locBeamP4_COM.Pz());

        // Fill Flat P4_1 branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_1_E", locProtonP4_COM.E());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_1_Px", locProtonP4_COM.Px());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_1_Py", locProtonP4_COM.Py());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_1_Pz", locProtonP4_COM.Pz());

        // Fill Flat P4_2 branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_2_E", locDecayingKShort1P4_COM.E());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_2_Px", locDecayingKShort1P4_COM.Px());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_2_Py", locDecayingKShort1P4_COM.Py());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_2_Pz", locDecayingKShort1P4_COM.Pz());

        // Fill Flat P4_3 branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_3_E", locDecayingKShort2P4_COM.E());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_3_Px", locDecayingKShort2P4_COM.Px());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_3_Py", locDecayingKShort2P4_COM.Py());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_3_Pz", locDecayingKShort2P4_COM.Pz());

        // Fill Flat P4_4 branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_4_E", locPiPlus1P4_COM.E());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_4_Px", locPiPlus1P4_COM.Px());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_4_Py", locPiPlus1P4_COM.Py());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_4_Pz", locPiPlus1P4_COM.Pz());

        // Fill Flat P4_5 branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_5_E", locPiMinus1P4_COM.E());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_5_Px", locPiMinus1P4_COM.Px());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_5_Py", locPiMinus1P4_COM.Py());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_5_Pz", locPiMinus1P4_COM.Pz());

        // Fill Flat P4_6 branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_6_E", locPiPlus2P4_COM.E());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_6_Px", locPiPlus2P4_COM.Px());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_6_Py", locPiPlus2P4_COM.Py());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_6_Pz", locPiPlus2P4_COM.Pz());

        // Fill Flat P4_7 branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_7_E", locPiMinus2P4_COM.E());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_7_Px", locPiMinus2P4_COM.Px());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_7_Py", locPiMinus2P4_COM.Py());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("P4_7_Pz", locPiMinus2P4_COM.Pz());


        // Fill Flat Missing Energy/Momentum^2 branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("ME", locMissingP4.E());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("MM2", locMissingP4.M2());

        // Fill Flat RFL branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("RFL1", locKShort1RFL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("RFL2", locKShort2RFL);

        // Fill Flat ChiSqDOF branch
        dFlatTreeInterface->Fill_Fundamental<Float_t>("ChiSqDOF", locChiSqDOF);

        // Fill Flat RF branch
        dFlatTreeInterface->Fill_Fundamental<Float_t>("RF", locDeltaT_RF);

        // Fill Flat Proton_Z branch
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_Z", locProtonX4_Measured.Z());

        // Fill Flat Proton detector branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_P", locProton_P);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_Theta", locProton_Theta);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_dEdx_CDC", locProton_dEdx_CDC);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_dEdx_CDC_integral", locProton_dEdx_CDC_integral);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_dEdx_FDC", locProton_dEdx_FDC);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_dEdx_ST", locProton_dEdx_ST);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_dEdx_TOF", locProton_dEdx_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_E_BCAL", locProton_E_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_E_FCAL", locProton_E_FCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_DeltaT_BCAL", locProton_DeltaT_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_DeltaT_TOF", locProton_DeltaT_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_DeltaT_FCAL", locProton_DeltaT_FCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_Beta_BCAL", locProton_Beta_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_Beta_TOF", locProton_Beta_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("Proton_Beta_FCAL", locProton_Beta_FCAL);

        // Fill Flat PiPlus1 detector branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_P", locPiPlus1_P);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_Theta", locPiPlus1_Theta);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_dEdx_CDC", locPiPlus1_dEdx_CDC);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_dEdx_CDC_integral", locPiPlus1_dEdx_CDC_integral);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_dEdx_FDC", locPiPlus1_dEdx_FDC);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_dEdx_ST", locPiPlus1_dEdx_ST);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_dEdx_TOF", locPiPlus1_dEdx_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_E_BCAL", locPiPlus1_E_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_E_FCAL", locPiPlus1_E_FCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_DeltaT_BCAL", locPiPlus1_DeltaT_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_DeltaT_TOF", locPiPlus1_DeltaT_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_DeltaT_FCAL", locPiPlus1_DeltaT_FCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_Beta_BCAL", locPiPlus1_Beta_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_Beta_TOF", locPiPlus1_Beta_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus1_Beta_FCAL", locPiPlus1_Beta_FCAL);

        // Fill Flat PiMinus1 detector branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_P", locPiMinus1_P);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_Theta", locPiMinus1_Theta);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_dEdx_CDC", locPiMinus1_dEdx_CDC);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_dEdx_CDC_integral", locPiMinus1_dEdx_CDC_integral);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_dEdx_FDC", locPiMinus1_dEdx_FDC);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_dEdx_ST", locPiMinus1_dEdx_ST);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_dEdx_TOF", locPiMinus1_dEdx_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_E_BCAL", locPiMinus1_E_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_E_FCAL", locPiMinus1_E_FCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_DeltaT_BCAL", locPiMinus1_DeltaT_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_DeltaT_TOF", locPiMinus1_DeltaT_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_DeltaT_FCAL", locPiMinus1_DeltaT_FCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_Beta_BCAL", locPiMinus1_Beta_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_Beta_TOF", locPiMinus1_Beta_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus1_Beta_FCAL", locPiMinus1_Beta_FCAL);

        // Fill Flat PiPlus2 detector branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_P", locPiPlus2_P);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_Theta", locPiPlus2_Theta);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_dEdx_CDC", locPiPlus2_dEdx_CDC);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_dEdx_CDC_integral", locPiPlus2_dEdx_CDC_integral);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_dEdx_FDC", locPiPlus2_dEdx_FDC);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_dEdx_ST", locPiPlus2_dEdx_ST);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_dEdx_TOF", locPiPlus2_dEdx_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_E_BCAL", locPiPlus2_E_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_E_FCAL", locPiPlus2_E_FCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_DeltaT_BCAL", locPiPlus2_DeltaT_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_DeltaT_TOF", locPiPlus2_DeltaT_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_DeltaT_FCAL", locPiPlus2_DeltaT_FCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_Beta_BCAL", locPiPlus2_Beta_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_Beta_TOF", locPiPlus2_Beta_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiPlus2_Beta_FCAL", locPiPlus2_Beta_FCAL);

        // Fill Flat PiMinus2 detector branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_P", locPiMinus2_P);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_Theta", locPiMinus2_Theta);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_dEdx_CDC", locPiMinus2_dEdx_CDC);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_dEdx_CDC_integral", locPiMinus2_dEdx_CDC_integral);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_dEdx_FDC", locPiMinus2_dEdx_FDC);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_dEdx_ST", locPiMinus2_dEdx_ST);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_dEdx_TOF", locPiMinus2_dEdx_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_E_BCAL", locPiMinus2_E_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_E_FCAL", locPiMinus2_E_FCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_DeltaT_BCAL", locPiMinus2_DeltaT_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_DeltaT_TOF", locPiMinus2_DeltaT_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_DeltaT_FCAL", locPiMinus2_DeltaT_FCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_Beta_BCAL", locPiMinus2_Beta_BCAL);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_Beta_TOF", locPiMinus2_Beta_TOF);
        dFlatTreeInterface->Fill_Fundamental<Float_t>("PiMinus2_Beta_FCAL", locPiMinus2_Beta_FCAL);

        // Fill Flat KShort detector branches
        dFlatTreeInterface->Fill_Fundamental<Float_t>("KShort1_Z", locDecayingKShort1X4.Z());
        dFlatTreeInterface->Fill_Fundamental<Float_t>("KShort2_Z", locDecayingKShort2X4.Z());

        Fill_FlatTree();
    } // End of Combo Loop

    return kTRUE;
}

void DSelector_phase_1::Finalize(void) {
    DSelector::Finalize();
}
