#ifndef DSelector_phase_2_h
#define DSelector_phase_2_h

#include <iostream>

#include "DSelector/DSelector.h"
#include "DSelector/DHistogramActions.h"
#include "DSelector/DCutActions.h"

#include "TH1D.h"
#include "TH2D.h"

class DSelector_phase_2 : public DSelector {

    public:

        DSelector_phase_2(TTree* locTree = NULL) : DSelector(locTree){}
        virtual ~DSelector_phase_2(){}

        void Init(TTree *tree);
        Bool_t Process(Long64_t entry);

    private:

        void Get_ComboWrappers(void);
        void Finalize(void);

        // CREATE REACTION-SPECIFIC PARTICLE ARRAYS

        // Step 0
        DParticleComboStep* dStep0Wrapper;
        DBeamParticle* dComboBeamWrapper;
        DChargedTrackHypothesis* dProtonWrapper;

        // Step 1
        DParticleComboStep* dStep1Wrapper;
        DKinematicData* dDecayingKShort1Wrapper;
        DChargedTrackHypothesis* dPiMinus1Wrapper;
        DChargedTrackHypothesis* dPiPlus1Wrapper;

        // Step 2
        DParticleComboStep* dStep2Wrapper;
        DKinematicData* dDecayingKShort2Wrapper;
        DChargedTrackHypothesis* dPiMinus2Wrapper;
        DChargedTrackHypothesis* dPiPlus2Wrapper;

    ClassDef(DSelector_phase_2, 0);
};

void DSelector_phase_2::Get_ComboWrappers(void) {
    // Step 0
    dStep0Wrapper = dComboWrapper->Get_ParticleComboStep(0);
    dComboBeamWrapper = static_cast<DBeamParticle*>(dStep0Wrapper->Get_InitialParticle());
    dProtonWrapper = static_cast<DChargedTrackHypothesis*>(dStep0Wrapper->Get_FinalParticle(2));

    // Step 1
    dStep1Wrapper = dComboWrapper->Get_ParticleComboStep(1);
    dDecayingKShort1Wrapper = dStep1Wrapper->Get_InitialParticle();
    dPiMinus1Wrapper = static_cast<DChargedTrackHypothesis*>(dStep1Wrapper->Get_FinalParticle(0));
    dPiPlus1Wrapper = static_cast<DChargedTrackHypothesis*>(dStep1Wrapper->Get_FinalParticle(1));

    // Step 2
    dStep2Wrapper = dComboWrapper->Get_ParticleComboStep(2);
    dDecayingKShort2Wrapper = dStep2Wrapper->Get_InitialParticle();
    dPiMinus2Wrapper = static_cast<DChargedTrackHypothesis*>(dStep2Wrapper->Get_FinalParticle(0));
    dPiPlus2Wrapper = static_cast<DChargedTrackHypothesis*>(dStep2Wrapper->Get_FinalParticle(1));

}

#endif // DSelector_phase_2_h
