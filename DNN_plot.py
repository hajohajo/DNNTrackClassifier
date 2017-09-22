import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from root_pandas import read_root


# Poor-man enum class with string conversion
class _Enum:
     def __init__(self, **values):
         self._reverse = {}
         for key, value in values.iteritems():
             setattr(self, key, value)
             if value in self._reverse:
                 raise Exception("Value %s is already used for a key %s, tried to re-add it for key %s" % (value, self._reverse[value], key))
             self._reverse[value] = key

     def toString(self, val):
         return self._reverse[val]


Algo = _Enum(
     undefAlgorithm = 0, ctf = 1,
     duplicateMerge = 2, cosmics = 3,
     initialStep = 4,
     lowPtTripletStep = 5,
     pixelPairStep = 6,
     detachedTripletStep = 7,
     mixedTripletStep = 8,
     pixelLessStep = 9,
     tobTecStep = 10,
     jetCoreRegionalStep = 11,
     conversionStep = 12,
     muonSeededStepInOut = 13,
     muonSeededStepOutIn = 14,
     outInEcalSeededConv = 15, inOutEcalSeededConv = 16,
     nuclInter = 17,
     standAloneMuon = 18, globalMuon = 19, cosmicStandAloneMuon = 20, 
cosmicGlobalMuon = 21,
     # Phase1
     highPtTripletStep = 22, lowPtQuadStep = 23, detachedQuadStep = 24,
     reservedForUpgrades1 = 25, reservedForUpgrades2 = 26,
     bTagGhostTracks = 27,
     beamhalo = 28,
     gsf = 29,
     # HLT algo name
     hltPixel = 30,
     # steps used by PF
     hltIter0 = 31,
     hltIter1 = 32,
     hltIter2 = 33,
     hltIter3 = 34,
     hltIter4 = 35,
     # steps used by all other objects @HLT
     hltIterX = 36,
     # steps used by HI muon regional iterative tracking
     hiRegitMuInitialStep = 37,
     hiRegitMuLowPtTripletStep = 38,
     hiRegitMuPixelPairStep = 39,
     hiRegitMuDetachedTripletStep = 40,
     hiRegitMuMixedTripletStep = 41,
     hiRegitMuPixelLessStep = 42,
     hiRegitMuTobTecStep = 43,
     hiRegitMuMuonSeededStepInOut = 44,
     hiRegitMuMuonSeededStepOutIn = 45,
     algoSize = 46
)




def efficiency(y_true,y_pred,goal,thresh):
	eff=np.zeros((len(thresh)))
	Ntot=(y_true==goal).sum()
	for value in range(0,len(thresh)):
		if value%200==0:
			print str(value)+"/"+str(len(thresh))
		Nsel=((y_true==goal) & (y_pred>thresh[value])).sum()
		eff[value]=1.0*Nsel/Ntot
	return eff

data=read_root('trackingNtuple_forPlotting.root','tree')
#data_pl=data_pl[:100000]

bins_DNN=np.linspace(0.0,1.0,num=10000,endpoint=True) #10000
bins_BDT=np.linspace(-1.0,1.0,num=20000,endpoint=True) #20000

cuts=[[-0.95,-0.75],
	[-0.65,-0.15],
	[0.2,0.4],
	[0.0,0.4],
	[-0.5,0.5],
	[-0.2,0.8],
	[-0.2,0.3],
	[-0.5,0.5],
	[-0.4,0.4],
	[-0.6,-0.3],
	[-0.2,0.4]]

looseHPBins=[]

for i in range(0,len(cuts)):
	step=2.0/len(bins_BDT)
	looseBin=int(round(abs(-1.0-cuts[i][0])/step))
	tightBin=int(round(abs(-1.0-cuts[i][1])/step))
	looseHPBins.append([looseBin,tightBin])

#print(looseHPBins)
#
#for i in range(0,len(looseHPBins)):
#	print bins_BDT[looseHPBins[i][0]],",",bins_BDT[looseHPBins[i][1]]
#
#print cuts

steps=[4,23,22,5,24,7,6,8,9,10,11]
ind=0

columns=[]
indices=['Loose','High purity']
impr_eff=[]
impr_rej=[]

Eff_DNN=[]
Eff_BDT=[]
Rej_DNN=[]
Rej_BDT=[]
fakerate_BDT=[]
fakerate_DNN=[]

for step in steps:
	data_pl=data[data['trk_algo'].values==step]

	truth=data_pl.trk_isTrue
	pred_DNN=data_pl.trk_mva_DNN
	pred_BDT=data_pl.trk_mva

	y_test=np.array(truth.iloc[:])
	y_pred_DNN=np.array(pred_DNN.iloc[:])
	y_pred_BDT=np.array(pred_BDT.iloc[:])

	#For fakerate calculation, to be multiplied by the eff
	N_fakes=(y_test==0).sum()
	N_trues=(y_test==1).sum()

	from sklearn import metrics

	true_eff_DNN=efficiency(y_test,y_pred_DNN,1,bins_DNN)
	fake_eff_DNN=efficiency(y_test,y_pred_DNN,0,bins_DNN)

	true_eff_BDT=efficiency(y_test,y_pred_BDT,1,bins_BDT)
	fake_eff_BDT=efficiency(y_test,y_pred_BDT,0,bins_BDT)

	roc_auc_DNN = round(metrics.auc(true_eff_DNN,np.subtract(1.0,fake_eff_DNN)),3)
	roc_auc_BDT = round(metrics.auc(true_eff_BDT,np.subtract(1.0,fake_eff_BDT)),3)

	#Calculate efficiencies for current cuts for plotting
	loose_true_trk_eff=true_eff_BDT[looseHPBins[ind][0]]
	hp_true_trk_eff=true_eff_BDT[looseHPBins[ind][1]]
	loose_fake_trk_rej_eff=1.0000-fake_eff_BDT[looseHPBins[ind][0]]
        hp_fake_trk_rej_eff=1.0000-fake_eff_BDT[looseHPBins[ind][1]]

	#Calculate improvements wrt to current selection
	true_eff_l_bin=0
	true_eff_hp_bin=0
	fake_rej_eff_l_bin=0
	fake_rej_eff_hp_bin=0
	error=0.0001
	for i in range(0,len(bins_DNN)):
		if(abs(true_eff_DNN[i]-loose_true_trk_eff)<error):
			true_eff_l_bin=i
		if(abs(true_eff_DNN[i]-hp_true_trk_eff)<error):
			true_eff_hp_bin=i
		if(abs((1.0000-fake_eff_DNN[i])-loose_fake_trk_rej_eff)<error):
			fake_rej_eff_l_bin=i
		if(abs((1.0000-fake_eff_DNN[i])-hp_fake_trk_rej_eff)<error):
			fake_rej_eff_hp_bin=i

#sameTrue
#	DNN_loose_true_trk_eff=true_eff_DNN[true_eff_l_bin]
#	DNN_hp_true_trk_eff=true_eff_DNN[true_eff_hp_bin]
	DNN_loose_fake_trk_rej_eff=1.0-fake_eff_DNN[true_eff_l_bin]
        DNN_hp_fake_trk_rej_eff=1.0-fake_eff_DNN[true_eff_hp_bin]

#sameFake
	DNN_loose_true_trk_eff=true_eff_DNN[fake_rej_eff_l_bin]
	DNN_hp_true_trk_eff=true_eff_DNN[fake_rej_eff_hp_bin]
#	DNN_loose_fake_trk_rej_eff=1.0-fake_eff_DNN[fake_rej_eff_l_bin]
#	DNN_hp_fake_trk_rej_eff=1.0-fake_eff_DNN[fake_rej_eff_hp_bin]

#	print "DNN_L bin: "+str(true_eff_l_bin)+"/"+str(len(bins_DNN))+" Trk_eff: "+str(true_eff_DNN[true_eff_l_bin])+" Value: "+str(1.0-fake_eff_DNN[true_eff_l_bin])
#	print "DNN_HP bin: "+str(true_eff_hp_bin)+"/"+str(len(bins_DNN))+" Trk_eff: "+str(true_eff_DNN[true_eff_hp_bin])+" Value: "+str(1.0-fake_eff_DNN[true_eff_hp_bin])

#	impr_true_trk_loose=(DNN_loose_true_trk_eff-loose_true_trk_eff)/loose_trk_eff
#	impr_fake_trk_rej_loose=(DNN_loose_fake_trk_rej_eff-loose_fake_trk_rej_eff)/loose_fake_trk_rej_eff
#	impr_true_trk_hp=(DNN_hp_true_trk_eff-hp_true_trk_eff)/hp_true_trk_eff
#	impr_fake_trk_rej_hp=(DNN_hp_fake_trk_rej_eff-hp_fake_trk_rej_eff)/hp_fake_trk_rej_eff

	#Various statistics for tables
	Eff_DNN.append([round(DNN_loose_true_trk_eff,3),round(DNN_hp_true_trk_eff,3)])
	Eff_BDT.append([round(loose_true_trk_eff,3),round(hp_true_trk_eff,3)])
	Rej_DNN.append([round(DNN_loose_fake_trk_rej_eff,3),round(DNN_hp_fake_trk_rej_eff,3)])
	Rej_BDT.append([round(loose_fake_trk_rej_eff,3),round(hp_fake_trk_rej_eff,3)])

	impr_true_trk_loose=(DNN_loose_true_trk_eff-loose_true_trk_eff)
	impr_fake_trk_rej_loose=(DNN_loose_fake_trk_rej_eff-loose_fake_trk_rej_eff)
	impr_true_trk_hp=(DNN_hp_true_trk_eff-hp_true_trk_eff)
	impr_fake_trk_rej_hp=(DNN_hp_fake_trk_rej_eff-hp_fake_trk_rej_eff)

	#Assume true trk eff is kept constant for DNN
	fakerate_BDT.append([round((1.0-loose_fake_trk_rej_eff)*N_fakes/((1.0-loose_fake_trk_rej_eff)*N_fakes+loose_true_trk_eff*N_trues),3),
				round((1.0-hp_fake_trk_rej_eff)*N_fakes/((1.0-hp_fake_trk_rej_eff)*N_fakes+hp_true_trk_eff*N_trues),3)])

        fakerate_DNN.append([round((1.0-DNN_loose_fake_trk_rej_eff)*N_fakes/((1.0-DNN_loose_fake_trk_rej_eff)*N_fakes+loose_true_trk_eff*N_trues),3),
                                round((1.0-DNN_hp_fake_trk_rej_eff)*N_fakes/((1.0-DNN_hp_fake_trk_rej_eff)*N_fakes+hp_true_trk_eff*N_trues),3)])

	#For comparison tables
	impr_eff.append([round(impr_true_trk_loose,3),round(impr_true_trk_hp,3)])
	impr_rej.append([round(impr_fake_trk_rej_loose,3),round(impr_fake_trk_rej_hp,3)])
	columns.append(Algo.toString(step))


#	plt.plot(bins_BDT[:-1],true_eff_BDT[:-1],label='BDT true')
#	plt.plot(bins_BDT[:-1],fake_eff_BDT[:-1],label='BDT fakes')
#	plt.ylim((0.0,1.2))
#	plt.xlim((-1.0,1.0))
#	plt.legend()
#	plt.show()

#plt.clf()

	plt.plot(true_eff_DNN,np.subtract(1.0,fake_eff_DNN),label='ROC curve DNN (area = %0.3f)' % roc_auc_DNN)
	plt.plot(true_eff_BDT,np.subtract(1.0,fake_eff_BDT),label='ROC curve BDT (area = %0.3f)' % roc_auc_BDT)
	plt.plot([0.0,loose_true_trk_eff],[loose_fake_trk_rej_eff,loose_fake_trk_rej_eff],'k--',label='BDT Loose cut')
	plt.plot([loose_true_trk_eff,loose_true_trk_eff],[0.0,loose_fake_trk_rej_eff],'k--')
        plt.plot([0.0,hp_true_trk_eff],[hp_fake_trk_rej_eff,hp_fake_trk_rej_eff],'k:',label='BDT High purity cut')
        plt.plot([hp_true_trk_eff,hp_true_trk_eff],[0.0,hp_fake_trk_rej_eff],'k:')

#        plt.plot([loose_true_trk_eff,DNN_loose_true_trk_eff],[loose_fake_trk_rej_eff,loose_fake_trk_rej_eff],'r--',label='DNN Loose cut')
#        plt.plot([loose_true_trk_eff,loose_true_trk_eff],[loose_fake_trk_rej_eff,DNN_loose_fake_trk_rej_eff],'r--')
#        plt.plot([hp_true_trk_eff,DNN_hp_true_trk_eff],[hp_fake_trk_rej_eff,hp_fake_trk_rej_eff],'r:',label='DNN High purity cut')
#        plt.plot([hp_true_trk_eff,hp_true_trk_eff],[hp_fake_trk_rej_eff,DNN_hp_fake_trk_rej_eff],'r:')

#        plt.plot([0.0,DNN_loose_true_trk_eff],[DNN_loose_fake_trk_rej_eff,DNN_loose_fake_trk_rej_eff],'r--',label='DNN Loose cut')
#        plt.plot([DNN_loose_true_trk_eff,DNN_loose_true_trk_eff],[0.0,DNN_loose_fake_trk_rej_eff],'r--')
#        plt.plot([0.0,DNN_hp_true_trk_eff],[DNN_hp_fake_trk_rej_eff,DNN_hp_fake_trk_rej_eff],'r:',label='DNN High purity cut')
#        plt.plot([DNN_hp_true_trk_eff,DNN_hp_true_trk_eff],[0.0,DNN_hp_fake_trk_rej_eff],'r:')


	plt.ylabel('Fake track rejection')
	plt.xlabel('True track selection efficiency')
	plt.ylim(0.0,1.1)
	plt.xlim(0.0,1.1)
	plt.title(Algo.toString(step))
	plt.legend()
#	plt.show()
	plt.savefig('plots/roc_'+Algo.toString(step)+'.pdf')

	plt.clf()

	ind+=1

	print "Algo: "+Algo.toString(step)
	print "True trk eff loose: "+str(loose_true_trk_eff)+" "+str(DNN_loose_true_trk_eff)
	print "True trk eff hp: "+str(hp_true_trk_eff)+" "+str(DNN_hp_true_trk_eff)
	print "Fake trk rej eff loose: "+str(loose_fake_trk_rej_eff)+" "+str(DNN_loose_fake_trk_rej_eff)
	print "Fake trk rej eff hp: "+str(hp_fake_trk_rej_eff)+" "+str(DNN_hp_fake_trk_rej_eff)

#table_eff=pd.DataFrame(np.transpose(impr_eff),index=indices,columns=columns)
#table_rej=pd.DataFrame(np.transpose(impr_rej),index=indices,columns=columns)
#print table_eff
#print table_rej

Eff_BDT=np.transpose(Eff_BDT)
Eff_DNN=np.transpose(Eff_DNN)
Rej_BDT=np.transpose(Rej_BDT)
Rej_DNN=np.transpose(Rej_DNN)
fakerate_BDT=np.transpose(fakerate_BDT)
fakerate_DNN=np.transpose(fakerate_DNN)

effies=np.core.defchararray.add(np.core.defchararray.add(Eff_BDT.astype(np.string_)," $\rightarrow$ "),Eff_DNN.astype(np.string_))
rejjies=np.core.defchararray.add(np.core.defchararray.add(Rej_BDT.astype(np.string_)," $\rightarrow$ "),Rej_DNN.astype(np.string_))
fakies=np.core.defchararray.add(np.core.defchararray.add(fakerate_BDT.astype(np.string_)," $\rightarrow$ "),fakerate_DNN.astype(np.string_)) 

table_eff=pd.DataFrame(effies,index=indices,columns=columns)
table_rej=pd.DataFrame(rejjies,index=indices,columns=columns)
table_fakerate=pd.DataFrame(fakies,index=indices,columns=columns)


outFile = open("Latex_tables.txt","w")
outFile.write("\n Improvement in true track efficiency \n")
outFile.write(table_eff.to_latex(escape=False))
outFile.write("\n Improvement in fake track rejection \n")
outFile.write(table_rej.to_latex(escape=False))
outFile.write("\n Improvement in fakerate \n")
outFile.write(table_fakerate.to_latex(escape=False))
outFile.close()

