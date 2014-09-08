#include "SimpleOpt.h"

#include "loadHistogram.hpp"
#include "loadFitResult.hpp"
#include "scanTau.hpp"
#include "logging.hpp"
#include "utils.hpp"

#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TROOT.h"
#include "TMath.h"
#include "TColor.h"
#include "TCanvas.h"
#include "TLegend.h"

#include "TUnfold.h"
#include "TUnfoldSys.h"

#include <iostream>
#include <cstdio>
#include <vector>
#include <cmath>
#include <map>
#include <set>
#include <array>
#include <string>
#include <fstream>
#include <cstdio>

static const std::string DATA="DATA";
static const std::string SIGNAL="tchan";
static const std::set<std::string> BACKGROUND={"ttjets","wzjets","qcd"};
static const std::string HISTPREFIX="2j1t_cos_theta_lj__";
static const unsigned int REBIN_RECO=12;
static const unsigned int REBIN_GEN=6;



void doUnfolding(const std::vector<std::string>& histFiles,
       const std::string& signalHistName,
       const std::set<std::string>& backgroundHistNames,
       const std::string& dataHistName,
       const std::vector<std::string>& responseFiles,
       const std::string& responseMatrixName,
       const std::string& fitResult,
       const std::string& fitCovariance,
       const std::string& outputFileName,
       const std::string& sys="nominal",
       bool noStatUnc=true,
       bool noMCStatUnc=true,
       bool noFitUnc=true,
       bool mcOnly=false,
       double fixedTau=-1.0)
{
	TH1::SetDefaultSumw2(true);


    FitResult fitFactors = loadFitResult(fitResult);
    
    
    TH1F* dataHist = nullptr;
    if (mcOnly)
    {
        dataHist = loadHistogram1D(histFiles,HISTPREFIX+signalHistName,"nominal",fitFactors[signalHistName][0],REBIN_RECO);
        for (const std::string& histName: backgroundHistNames)
        {
            //nice mem leak
            dataHist->Add(loadHistogram1D(histFiles,HISTPREFIX+histName,"nominal",fitFactors[histName][0],REBIN_RECO));
        }
        for (int i=0; i<dataHist->GetNbinsX()+2; ++i)
        {
            //set statistical uncertainty to Poisson
            dataHist->SetBinError(i,sqrt(dataHist->GetBinContent(i)));
        }
    }
    else
    {
        dataHist = loadHistogram1D(histFiles,HISTPREFIX+dataHistName,"nominal",1.0,REBIN_RECO);
    }
    
    log(INFO,"data hist: %i bins, N=%f\n",dataHist->GetNbinsX(),dataHist->Integral());
    
    TH2F* responseHist = loadHistogram2D(responseFiles,responseMatrixName,sys,fitFactors[signalHistName][0],REBIN_GEN,REBIN_RECO);
    log(INFO,"response hist: %ix%i, N=%f\n",responseHist->GetNbinsX(),responseHist->GetNbinsY(),responseHist->Integral());
    
    
    TUnfoldSys tunfold(responseHist,TUnfold::kHistMapOutputHoriz,TUnfold::kRegModeCurvature);
    tunfold.SetBias(responseHist->ProjectionX());
    tunfold.SetInput(dataHist);
    
    //TODO: decorrelate bgs first!
    for (FitResult::const_iterator it = fitFactors.begin(); it != fitFactors.end(); ++it)
    {
        if (backgroundHistNames.find(it->first)!=backgroundHistNames.end())
        {
            TH1F* bgHist = loadHistogram1D(histFiles,HISTPREFIX+it->first,sys,1.0,REBIN_RECO);
            tunfold.SubtractBackground(bgHist,it->first.c_str(),fitFactors[it->first][0],fitFactors[it->first][1]);
        }
    }
    
    TFile outputFile(outputFileName.c_str(),"RECREATE");
    
    
    if (fixedTau>0.0)
    {
        tunfold.DoUnfold(fixedTau,dataHist,1.0);
    }
    else
    {
        ScanResult scanResult = scanTau(responseHist,dataHist);
        log(INFO,"regularization scan gives: tau=%5.4e @ rho=%4.3f\n",scanResult.taumean,scanResult.pmean);
        tunfold.DoUnfold(scanResult.taumean,dataHist,1.0);
    }
    TH1F *unfoldedHist = new TH1F("unfolded","unfolded", REBIN_GEN, -1, 1);
	tunfold.GetOutput(unfoldedHist);
	unfoldedHist->Write();
    std::cout << "A=" << asymmetry(unfoldedHist) << std::endl;	
	
	TH2D *hrhoij = new TH2D("correlation","correlation",REBIN_GEN,1,REBIN_GEN,REBIN_GEN,1,REBIN_GEN);
	tunfold.GetRhoIJ(hrhoij);
	hrhoij->Write();
	TH1D *hrhoi = new TH1D("1dcorr","1dcorr",REBIN_GEN,1,REBIN_GEN);
	tunfold.GetRhoI(hrhoi);
	hrhoi->Write();
	responseHist->ProjectionX("gen")->Write();

	TH2D* errorMatrix = new TH2D("error","error",REBIN_GEN,1,REBIN_GEN,REBIN_GEN,1,REBIN_GEN);
	if (!noStatUnc)
	{
	    //add error through input vector (aka statistical uncertainty)
	    tunfold.GetEmatrixInput(errorMatrix, 0, false);
	}
	if (!noMCStatUnc)
	{
	    //add error through limited MC of response matrix
	    tunfold.GetEmatrixSysUncorr(errorMatrix ,0, false);  
	    for (FitResult::const_iterator it = fitFactors.begin(); it != fitFactors.end(); ++it)
        {
            if (backgroundHistNames.find(it->first)!=backgroundHistNames.end())
            {
	            //add error through limited MC of subtracted background histograms
	            tunfold.GetEmatrixSysBackgroundUncorr(errorMatrix,it->first.c_str() ,0, false);
            }
        } 
	}
    if (!noFitUnc)
    {
        for (FitResult::const_iterator it = fitFactors.begin(); it != fitFactors.end(); ++it)
        {
            if (backgroundHistNames.find(it->first)!=backgroundHistNames.end())
            {
                //add error through fit of background normalization
	            tunfold.GetEmatrixSysBackgroundScale(errorMatrix,it->first.c_str(), 0, false); 
	        }
        }
	}
	errorMatrix->Write();
	
	outputFile.Close();
}




enum {
    OPT_SYS, 
    OPT_HISTFILE,
    OPT_FITRESULTPREFIX, 
    OPT_FITRESULT, 
    OPT_FITCOVARIANCE, 
    OPT_RESPONSEFILE,
    OPT_RESPONSEMATRIXNAME,
    OPT_OUTPUT,
    OPT_NOSTAT,
    OPT_NOMC,
    OPT_NOFITERROR,
    OPT_MCONLY,
    OTP_FIXEDTAU,
    OTP_SCANTAU
};

CSimpleOpt::SOption options[] ={
    {OPT_SYS, "--sys", SO_REQ_SEP},
    {OPT_HISTFILE,"--histFile",SO_REQ_SEP},
    {OPT_FITRESULTPREFIX,"--fitResultPrefix",SO_REQ_SEP},
    {OPT_FITRESULT,"--fitResult",SO_REQ_SEP},
    {OPT_FITCOVARIANCE,"--fitCovariance",SO_REQ_SEP},
    {OPT_RESPONSEFILE,"--responseFile",SO_REQ_SEP},
    {OPT_RESPONSEMATRIXNAME,"--responseMatrixName",SO_REQ_SEP},
    {OPT_OUTPUT,"--output",SO_REQ_SEP},
    {OPT_NOSTAT,"--no-stat",SO_NONE},
    {OPT_NOMC,"--no-mcstat",SO_NONE},
    {OPT_NOFITERROR,"--no-fiterror",SO_NONE},
    {OPT_MCONLY,"--mc-only",SO_NONE},
    {OTP_FIXEDTAU,"--fixedTau",SO_REQ_SEP},
    {OTP_SCANTAU,"--scanTau",SO_NONE},
    SO_END_OF_OPTIONS

};

int main(int argc, char* argv[])
{
    CSimpleOpt parser(argc,argv,options);
    
    std::string systematic = "nominal";
    std::string fitResultPrefix="";
    std::string fitResultFile="mu.txt";
    std::string covarianceFile="mu_cov.root";
    std::vector<std::string> histFiles;
    std::vector<std::string> responseFiles;
    std::string responseMatrixName = "tm";
    std::string outputFile = "out.root";
    
    bool noStat = false;
    bool noMCStat = false;
    bool noFitError = false;
    bool mcOnly = false;
    
    double fixedTau=-1.0;
    
    gErrorIgnoreLevel = kPrint | kInfo | kWarning;
    
    setupLogging();
    
    while (parser.Next())
    {
        if (parser.LastError() == SO_SUCCESS)
        {
            if (parser.OptionId() == OPT_SYS) 
            {
                systematic=parser.OptionArg();
            }
            else if (parser.OptionId() == OPT_HISTFILE) 
            {
                histFiles.push_back(parser.OptionArg());
            }
            else if (parser.OptionId() == OPT_FITRESULTPREFIX) 
            {
                fitResultPrefix = parser.OptionArg();
            }
            else if (parser.OptionId() == OPT_FITRESULT) 
            {
                fitResultFile = parser.OptionArg();
            }
            else if (parser.OptionId() == OPT_FITCOVARIANCE) 
            {
                covarianceFile = parser.OptionArg();
            }
            else if (parser.OptionId() == OPT_RESPONSEFILE)
            {
                responseFiles.push_back(parser.OptionArg());
            }
            else if (parser.OptionId() == OPT_RESPONSEMATRIXNAME)
            {
                responseMatrixName=parser.OptionArg();
            }
            else if (parser.OptionId() == OPT_OUTPUT)
            {
                outputFile  = parser.OptionArg();
            }
            else if (parser.OptionId() ==OPT_NOSTAT)
            {
                noStat=true;
            }
            else if (parser.OptionId() ==OPT_NOMC)
            {
                noMCStat=true;
            }
            else if (parser.OptionId() ==OPT_NOFITERROR)
            {
                noFitError = true;
            }
            else if (parser.OptionId() ==OPT_MCONLY)
            {
                mcOnly=true;
            }
            else if (parser.OptionId() ==OTP_FIXEDTAU)
            {
                try
                {
                    fixedTau=stod(std::string(parser.OptionArg()));
                } 
                catch (std::exception e)
                {
                    log(ERROR,"while parsing the given fixed tau: %s\n",e.what());
                }
            }
            else if (parser.OptionId() ==OTP_SCANTAU)
            {
                fixedTau=-1.0; //neg value will trigger a scan
            }   
        }
        else
        {
            log(CRITICAL,"Invalid argument: %s\n", parser.OptionText());
            return 1;
        }
    }
    
    
    
    log(INFO,"histfiles: \n");
    for (const std::string& histFile: histFiles)
    {
        log(INFO," ... %s\n",histFile.c_str());
    }
    
    log(INFO,"responseFiles: \n");
    for (unsigned int i = 0; i< responseFiles.size();++i)
    {
        log(INFO," ... %s\n",responseFiles[i].c_str());
    }
    
    log(INFO,"matrix name: %s\n",(responseMatrixName+"__"+systematic).c_str());
    
    log(INFO,"systematic:\n ... %s\n",systematic.c_str());
    
    log(INFO,"fit:\n");
    log(INFO," ... scale: %s\n",(fitResultPrefix+"/"+systematic+"/"+fitResultFile).c_str());
    log(INFO," ... covariance: %s\n",(fitResultPrefix+"/"+systematic+"/"+covarianceFile).c_str());

    if (fixedTau<0.0)
    {
        log(INFO,"will scan for optimal regularization\n");
    }
    else
    {
        log(INFO,"use fixed regularization: %e\n",fixedTau);
    }
    
    doUnfolding(
        histFiles,
        SIGNAL,
        BACKGROUND,
        DATA,
        responseFiles,
        responseMatrixName,
        fitResultPrefix+"/"+systematic+"/"+fitResultFile,
        fitResultPrefix+"/"+systematic+"/"+covarianceFile,
        outputFile,
        systematic,
        noStat, //stat unc
        noMCStat, //mcstat unc
        noFitError, //fitunc
        mcOnly, //only mc
        fixedTau
    );
    
    
    return 0;
}
