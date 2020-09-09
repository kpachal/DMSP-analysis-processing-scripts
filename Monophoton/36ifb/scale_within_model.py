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
parser.add_argument('path', metavar='path', type=str, nargs='+',
                    help='the path with the json files to process')
parser.add_argument('--originals', nargs='?', type=str,
                    help='directory with original contours for certain models and couplings to use for validation')

args = parser.parse_args()
directory = args.path[0]
if not os.path.exists(directory) :
  print("Path",directory,"does not exist!")
  exit(1)

# Target models.
# Using this method you can ONLY scale to targets in the same
# model as your input, so there is no "model" parameter available.
targets = [{"gq" : 0.1, "gDM" : 1.0, "gl" : 0.1}, {"gq" : 0.1, "gDM" : 1.0, "gl" : 0.0}]

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

# Start.
# Exit if things are not as expected.
jsons = glob.glob(directory+"/*.json")
if not len(jsons)==2 :
  print("You need to provide exactly 2 json files: one with the HEPData to be analysed and one specifying the parameter models of that HEPdata.")
  exit(1)
config = None
for jfile in [i.split("/")[-1] for i in jsons] : 
  with open(jfile, "r") as read_file :
    if jfile == "info.json" :
      config = json.load(read_file)
    else :
      data = json.load(read_file)

# Now begin extracting data.
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
paper_scenario = {"gq" : config["gq"], "gDM" : config["gDM"], "gl" : config["gl"]}
paper_scenario["xvals"] = xlist
paper_scenario["yvals"] = ylist
paper_scenario["zvals"] = zlist

def get_scenario_name(model, gq, gDM, gl) :
  scenario_name = model+"_gq{0}_gchi{1}_gl{2}".format(gq,gDM,gl).replace(".","p")
  return scenario_name

def get_plot_text(model, gq, gDM, gl) :
  use_text = "{0}\ng$_{4}$={1}, g$_{5}$={2}, g$_{6}$={3}".format(("Axial-vector" if model=="AV" else "Vector"),gq, gDM, gl,"q","\chi","l")
  return use_text

# Make a contour plot
from common_plotter import make_plot
paper_name = get_scenario_name(config["model"], config["gq"], config["gDM"], config["gl"])
paper_text = get_plot_text(config["model"], config["gq"], config["gDM"], config["gl"])
make_plot(xlist,ylist,zlist,config["analysis_tag"]+"_original",plot_limits=plot_limits,addText="Original\n"+paper_text,addPoints=drawPoints,interp_method=interp_method,do_mirror=do_mirror,do_rotate=do_rotate,doContour=doContour)

# Now convert to each of our other scenarios and let's see how plausible it looks
from monojet_functions import *

# Conversion - scale by xsec(old)/xsec(new) bc theory_new is on the bottom
for scenario in targets :

  scenario_name = get_scenario_name(config["model"],scenario["gq"],scenario["gDM"],scenario["gl"])

  #print("Beginning scenario",scenario)
  start = time.perf_counter()

  converted_x = []
  converted_y = []
  converted_z = []

  # paper_val = sigma_obs/sigma_theory
  # Want to convert to sigma_obs/sigma_new_theory
  # So multiply by (sigma_theory/sigma_new_theory)
  nPointsKept = 0
  for (mMed, mDM, paper_val) in zip(xlist,ylist,zlist) :

    # Various escape conditions
    if mMed > point_limits[0] or mDM > point_limits[1] : 
      continue
    nPointsKept = nPointsKept+1
    if doTest and nPointsKept > 5 :
      continue

    # Propagator only version.
    # Therefore must be the same model as parent.
    if config["model"] == "AV" :
      paper_xsec = relative_monox_propagator_integral_axial(mMed, mDM, config["gq"], config["gDM"], config["gl"])
      this_xsec = relative_monox_propagator_integral_axial(mMed, mDM, scenario["gq"], scenario["gDM"], scenario["gl"])
    else :
      paper_xsec = relative_monox_propagator_integral_vector(mMed, mDM, config["gq"], config["gDM"], config["gl"])
      this_xsec = relative_monox_propagator_integral_vector(mMed, mDM, scenario["gq"], scenario["gDM"], scenario["gl"])   

    if this_xsec == 0 :
      continue
    else :
      converted_val = paper_val*(paper_xsec/this_xsec)
      converted_z.append(converted_val)
      converted_x.append(mMed)
      converted_y.append(mDM)

  stop = time.perf_counter()
  print("Finished in {0} seconds.".format(round(stop-start)))

  # Now ready to make a plot.
  array_x = np.array(converted_x)
  array_y = np.array(converted_y)
  array_z = np.array(converted_z)
  text_here = get_plot_text(config["model"], scenario["gq"],scenario["gDM"],scenario["gl"])
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
for scenario in [paper_scenario]+targets :

  scenario_name = get_scenario_name(config["model"],scenario["gq"],scenario["gDM"],scenario["gl"])
  if not scenario_name in comparisons.keys() :
    continue

  print("Making comparison plot for scenario",scenario_name)
  comparison = comparisons[scenario_name]

  # Make patch from path
  patch = mpatches.PathPatch(comparison, edgecolor="red", facecolor=None, fill=False, lw=2)

  # And plot
  text_here = get_plot_text(config["model"], scenario["gq"], scenario["gDM"], scenario["gl"])
  if scenario_name == paper_name :
    text_here = "Original\n"+text_here
  else :
    text_here = "Rescaled\n"+text_here
  make_plot(scenario["xvals"], scenario["yvals"], scenario["zvals"],config["analysis_tag"]+"_"+scenario_name+"_compare"+plot_tag,plot_limits=plot_limits,addText=text_here, addCurves=[patch],addPoints=drawPoints,interp_method=interp_method,do_mirror=do_mirror,do_rotate=do_rotate,doContour=doContour)