Changes needed in the step3:

##
outputdir=sys.argv[5]
step=sys.argv[4]

if outputdir=="outputRoots_trackingNtuples":
        from Validation.RecoTrack.customiseTrackingNtuple import customiseTrackingNtuple
        process = customiseTrackingNtuple(process)
        process.TFileService = cms.Service("TFileService",
                fileName = cms.string("./"+outputdir+"/trackingNtuple_"+step+"_"+sys.argv[3]+".root")
        )
        process.trackingNtuple.tracks = step+"Tracks"
        process.trackingNtuple.trackMVAs = [step] #+"Classifier1"]
        process.trackingNtuple.vertices = "firstStepPrimaryVertices"
##
