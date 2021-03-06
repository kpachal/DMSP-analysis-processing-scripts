import json
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import scipy as sp
import time
import argparse
import glob
import os.path
import math

# Collect target directory from command line
parser = argparse.ArgumentParser(description='Propagator-based scaling to other couplings in the same model.')
parser.add_argument('file', metavar='file', type=str, nargs='+',
                    help='the json file to process and from which to perform the scaling. Note: if this json is not made from an earlier scaling, you must also include an info.json file in the directory specifying its contents.')
parser.add_argument('--originals', nargs='?', type=str,
                    help='directory with original contours for certain models and couplings to use for validation')

args = parser.parse_args()
target_json = args.file[0]
if not os.path.exists(target_json) :
  print(target_json,"does not exist!")
  exit(1)

# Target models.
# Using this method you can ONLY scale to targets in the same
# model as your input, so there is no "model" parameter available.
targets = [{"gq" : 0.1, "gDM" : 1.0, "gl" : 0.1}, {"gq" : 0.1, "gDM" : 1.0, "gl" : 0.0}, {"gq" : 0.1, "gDM" : 1.0, "gl" : 0.01}]

# To add (Boyu?)
# Add a 2D scan of g_q excluded between endpoints
# Would like to see your code for that!

# Add markers to the plots for the data points used?
drawPoints=True

# Draw our own contours through them?
doContour=False

# Set up which interpolation method to use.
# Options: griddata, rbf, clough_tocher, smooth_bivariate, lsq_bivariate, interp2d
interp_method = "griddata"

# These can help with interpolation stability
do_mirror = True
do_rotate = False

# Add this tag to all plot outputs
plot_tag = ""

# Don't bother scaling stuff outside here, it won't matter.
# [x, y]
point_limits = [3000,1500]

# Set this to enforce specific axis limits.
# Otherwise, will be set by last point value.
plot_limits = []
#plot_limits = [2600, 1100]

# This is for benchmarking. If I just want to test the time taken
# to do a few points and check everything runs, turn this on.
doTest = False

# Add validation to targets when available?
addValidation = False

# Start reading json files.
# Exit if things are not as expected.
# Two things we can run on:

# First, the info file.
# We need all the info in here if we're running off a HEPData table, but we
# will still pick up the analysis label otherwise.
directory = "/".join(target_json.split("/")[0:-1])
if not directory : directory = "."
json_basename = target_json.split("/")[-1]
info_file = directory+"/info.json" 
if not os.path.isfile(info_file) :
  print("Couldn't find an info.json file to match the requested json!")
  print("You need to include one when running on a table from HEPData.")
  print("Tried:",info_file)
  exit(1)

# Then we have one. Process it
with open(info_file, "r") as read_file :
  config = json.load(read_file)

# Pick up the target json.
# Here's the option for a file from HEPData.
is_original = False
if not "rescaled" in json_basename :
  is_original = True

  # Verify that the file listed in it is the one we've been asked to run on
  hepdatafile_path = config["hepdata_file"] if config["hepdata_file"].startswith("/") else directory+"/"+config["hepdata_file"]
  if not config["hepdata_file"].split("/")[-1] == json_basename :
    print("This info.json file doesn't seem to correspond to the json you've requested to run on. Please check!")
    exit(1)

  # It is: collect our data
  with open(hepdatafile_path, "r") as read_file :
    data = json.load(read_file)

  # Now begin extracting data.
  # This process works for the mono-photon HEPData at least. It's simpler for rescaled files.
  values = data["values"]
  # Extract as numpy arrays.
  # Usually mDM is the y axis, so if not it should be specified in the config.
  if not "DMaxis" in config.keys() or config["DMaxis"]=="y" :
    iDM = 1
    iMed = 0
  else :
    iDM = 0
    iMed = 1
  xlist = np.array([val["x"][iMed]["value"] for val in values]).astype(np.float)
  ylist = np.array([val["x"][iDM]["value"] for val in values]).astype(np.float)
  zlist = np.array([val["y"][0]["value"] for val in values]).astype(np.float)

  # Save in the same format we'll use for our converted scenarios, for ease of plotting.
  initial_scenario = {"model" : config["model"], "gq" : config["gq"], "gDM" : config["gDM"], "gl" : config["gl"]}
  initial_scenario["xvals"] = xlist
  initial_scenario["yvals"] = ylist
  initial_scenario["zvals"] = zlist

# Second, a file produced from an earlier reinterpretation.
# In this case, it contains its own model info and is already properly formatted.
else :

  with open(target_json,"r") as read_file :
    initial_scenario = json.load(read_file)

  xlist = np.array(initial_scenario["xvals"])
  ylist = np.array(initial_scenario["yvals"])
  zlist = np.array(initial_scenario["zvals"])

  # And now reassign them, because we need arrays:
  initial_scenario["xvals"] = xlist
  initial_scenario["yvals"] = ylist
  initial_scenario["zvals"] = zlist
  

def get_scenario_name(model, gq, gDM, gl) :
  scenario_name = model+"_gq{0}_gchi{1}_gl{2}".format(gq,gDM,gl).replace(".","p")
  return scenario_name

def get_plot_text(model, gq, gDM, gl) :
  use_text = "{0}\ng$_{4}$={1}, g$_{5}$={2}, g$_{6}$={3}".format(("Axial-vector" if model=="AV" else "Vector"),gq, gDM, gl,"q","\chi","l")
  return use_text

# Make a contour plot
from common_plotter import make_plot
initial_name = get_scenario_name(initial_scenario["model"], initial_scenario["gq"], initial_scenario["gDM"], initial_scenario["gl"])
initial_text = get_plot_text(initial_scenario["model"], initial_scenario["gq"], initial_scenario["gDM"], initial_scenario["gl"])
# If this is straight from the paper, make a plot with the "Original" label.
if is_original :
  make_plot(xlist,ylist,zlist,config["analysis_tag"]+"_original",plot_limits=plot_limits,addText="Original\n"+initial_text,addPoints=drawPoints,interp_method=interp_method,do_mirror=do_mirror,do_rotate=do_rotate,doContour=doContour)
# Otherwise, plot it but with "rescaled"
else :
  make_plot(xlist,ylist,zlist,config["analysis_tag"]+"_rescaled",plot_limits=plot_limits,addText="Original\n"+initial_text,addPoints=drawPoints,interp_method=interp_method,do_mirror=do_mirror,do_rotate=do_rotate,doContour=doContour)

# Now convert to each of our other scenarios and let's see how plausible it looks
from monojet_functions import *

# Conversion - scale by xsec(old)/xsec(new) bc theory_new is on the bottom
for scenario in targets :

  scenario_name = get_scenario_name(initial_scenario["model"],scenario["gq"],scenario["gDM"],scenario["gl"])

  #print("Beginning scenario",scenario)
  start = time.perf_counter()

  converted_x = []
  converted_y = []
  converted_z = []

  # initial_val = sigma_obs/sigma_theory
  # Want to convert to sigma_obs/sigma_new_theory
  # So multiply by (sigma_theory/sigma_new_theory)
  nPointsKept = 0
  for (mMed, mDM, initial_val) in zip(xlist,ylist,zlist) :

    # Various escape conditions
    if mMed > point_limits[0] or mDM > point_limits[1] : 
      continue
    nPointsKept = nPointsKept+1
    if doTest and nPointsKept > 5 :
      continue

    # Propagator only version.
    # Therefore must be the same model as parent.
    if initial_scenario["model"] == "AV" :
      initial_xsec = relative_monox_propagator_integral_axial(mMed, mDM, initial_scenario["gq"], initial_scenario["gDM"], initial_scenario["gl"])
      this_xsec = relative_monox_propagator_integral_axial(mMed, mDM, scenario["gq"], scenario["gDM"], scenario["gl"])
    else :
      initial_xsec = relative_monox_propagator_integral_vector(mMed, mDM, initial_scenario["gq"], initial_scenario["gDM"], initial_scenario["gl"])
      this_xsec = relative_monox_propagator_integral_vector(mMed, mDM, scenario["gq"], scenario["gDM"], scenario["gl"])   

    if this_xsec == 0 :
      continue
    else :
      converted_val = initial_val*(initial_xsec/this_xsec)
      converted_z.append(converted_val)
      converted_x.append(mMed)
      converted_y.append(mDM)

  stop = time.perf_counter()
  print("Finished in {0} seconds.".format(round(stop-start)))

  # Now ready to make a plot.
  array_x = np.array(converted_x)
  array_y = np.array(converted_y)
  array_z = np.array(converted_z)
  text_here = get_plot_text(initial_scenario["model"], scenario["gq"],scenario["gDM"],scenario["gl"])
  make_plot(array_x,array_y,array_z,config["analysis_tag"]+"_"+scenario_name+plot_tag,plot_limits=plot_limits,addText=text_here,addPoints=drawPoints,interp_method=interp_method,do_mirror=do_mirror,do_rotate=do_rotate,doContour=doContour)

  # and save.
  targets[targets.index(scenario)]["xvals"] = array_x
  scenario["yvals"] = array_y
  scenario["zvals"] = array_z

# Compare each of our contours to official ones if validations available.
# Otherwise we can just stop here.
if not args.originals :
  exit(0)

# So we have comparisons. Read in what we have and put them into plottable paths.
from matplotlib.path import Path
comparison_files = glob.glob(directory+"/"+args.originals+"/*.json")
comparisons = {}
for infile in comparison_files :
  scenario_keys = infile.split("_")
  if "AV" in scenario_keys : model = "AV"
  else : model = "V"
  scenario_name = "_".join(scenario_keys[scenario_keys.index(model):]).strip(".json")
  with open(infile, "r") as read_file :
    comparison_data = json.load(read_file)
  # Now make this into a path
  vertices = []
  for item in comparison_data["values"] : 
    vertices.append((item["x"][0]["value"],item["y"][0]["value"]))
  this_path = mpath.Path(vertices)
  comparisons[scenario_name] = this_path

print("Beginning to plot comparisons.")
for index,scenario in enumerate([initial_scenario]+targets) :

  scenario_name = get_scenario_name(initial_scenario["model"],scenario["gq"],scenario["gDM"],scenario["gl"])
  if not scenario_name in comparisons.keys() :
    continue

  print("Making comparison plot for scenario",scenario_name)
  comparison = comparisons[scenario_name]

  # Make patch from path
  patch = mpatches.PathPatch(comparison, edgecolor="red", facecolor=None, fill=False, lw=2)

  # And plot
  text_here = get_plot_text(initial_scenario["model"], scenario["gq"], scenario["gDM"], scenario["gl"])
  if index==0 and is_original :
    text_here = "Original\n"+text_here
  else :
    text_here = "Rescaled\n"+text_here
  make_plot(scenario["xvals"], scenario["yvals"], scenario["zvals"],config["analysis_tag"]+"_"+scenario_name+"_compare"+plot_tag,plot_limits=plot_limits,addText=text_here, addCurves=[patch],addPoints=drawPoints,interp_method=interp_method,do_mirror=do_mirror,do_rotate=do_rotate,doContour=doContour)