import glob
import json
import ROOT
import uproot

path = "/eos/atlas/atlascerngroupdisk/phys-exotics/jdm/monojet/ntuples/R20/Final_36ifb/DM_limits_36ifb/CL95/DMA/"

# Functionize so I can try with and without scaled values
def extract_values(filelist,branch) :

  # JSON for output
  datapoints_original = {}

  for file in filelist :
    # Parse masses and couplings
    filename = (file.split("/")[-1]).replace(".root","")
    tokens = filename.split("_")
    mDM = tokens[1]
    M = tokens[2]
    gq = tokens[3].replace("gq","").replace("p",".")
    tree = uproot.open(file)["stats"]
    vals = tree.array(branch)

    if not gq in datapoints_original.keys() :
      datapoints_original[gq] = []
    this_dict = {"x" : M, "y" : mDM, "value" : "{0}".format(vals[0])}
    datapoints_original[gq].append(this_dict)

  return datapoints_original

# Parse root files
files_original = glob.glob(path+"SimplifiedShapeFit_DMA_*_obs_CL95/*.root")
files_scaled = glob.glob(path+"scaled_SimplifiedShapeFit_DMA_*_obs_CL95/*.root")

# Get values
#dict_original = extract_values(files_original,"mu_hat_obs")
dict_original = extract_values(files_original,"obs_upperlimit")
dict_scaled = extract_values(files_scaled,"obs_upperlimit")

# Add them together
dict_total = {"0.25" : []}
for item in dict_original["0.25"] :
  dict_total["0.25"].append(item)
for item in dict_scaled["0.25"] :
  dict_total["0.25"].append(item)

# Write
with open("original_limits_AV_gq0p25_gchi1p0.json", "w") as data_file:
  json.dump(dict_total, data_file, indent=2)



