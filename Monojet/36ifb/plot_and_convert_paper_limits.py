import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import scipy as sp

# Analysing results from 
# https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/EXOT-2016-27/

analysis_tag = "EXOT-2016-27"
scenario_tag = "AV_gq0p25_gchi1p0"
paper_scenario = {"model" : "AV", "gq"  : 0.25, "gDM" : 1.0, "gl" : 0.0}

def get_aspect_ratio(ax) :
  ratio = 1.0
  xleft, xright = ax.get_xlim()
  ybottom, ytop = ax.get_ylim()
  return abs((xright-xleft)/(ybottom-ytop))*ratio

def make_plot(xvals, yvals, zvals, this_tag, addCurves=None) :

  levels = range(26)  # Levels must be increasing.
  fig,ax=plt.subplots(1,1)
  plt.xlim(0, 2600)
  plt.ylim(0, 1100)
  ratio = get_aspect_ratio(ax)
  ax.set_aspect(ratio)
  cp = ax.tricontourf(xvals, yvals, zvals, levels=levels, cmap='Blues_r')
  fig.colorbar(cp)
  ax.tricontour(xvals, yvals, zvals,levels=[1],colors=['w'],linewidths=[2])
  ax.set_xlabel("m$_{ZA}$ [GeV]")
  ax.set_ylabel("m$_{\chi}$ [GeV]")

  if addCurves :
    for curve in addCurves :
      ax.add_patch(curve)

  plt.savefig('{0}_{1}.eps'.format(analysis_tag,this_tag))
  plt.savefig('{0}_{1}.pdf'.format(analysis_tag,this_tag))

with open("hepdata_{0}.json".format(scenario_tag), "r") as read_file:
  data = json.load(read_file)

values = data["values"]

# Extract as numpy arrays
xlist = np.array([val["x"][0]["value"] for val in values]).astype(np.float)
ylist = np.array([val["x"][1]["value"] for val in values]).astype(np.float)
zlist = np.array([val["y"][0]["value"] for val in values]).astype(np.float)

# Make a contour plot matching the one from the analysis
make_plot(xlist,ylist,zlist,scenario_tag+"_original")

# Now convert to each of our other scenarios and let's see how plausible it looks
from monojet_functions import *
scenarios = {
  "AV_gq0p25_gchi1p0" : {"model" : "AV", "gq"  : 0.25, "gDM" : 1.0, "gl" : 0.0},
  "AV_gq0p1_gl0p1_gchi1p0" : {"model" : "AV", "gq" : 0.1, "gDM" : 1.0, "gl" : 0.1},
  "V_gq0p25_gchi1p0" : {"model" : "V", "gq"  : 0.25, "gDM" : 1.0, "gl" : 0.0},
  "V_gq0p1_gl0p01_gchi1p0" : {"model" : "V", "gq" : 0.1, "gDM" : 1.0, "gl" : 0.01}
}

# Conversion - scale by xsec(old)/xsec(new) bc theory_new is on the bottom?
for scenario in scenarios.keys() :

  this_scenario = scenarios[scenario]
  converted_x = []
  converted_y = []
  converted_z = []

  # Validation: pick one very on shell point and one off-shell point
  print("Validation, scenario",scenario)
  onshell_mMed = 1400
  onshell_mDM = 400
  offshell_mMed = 100
  offshell_mDM = 75
  paper_onshell_simple = xsec_monojet_axial(onshell_mMed, onshell_mDM, paper_scenario["gq"], paper_scenario["gDM"], paper_scenario["gl"])
  if this_scenario["model"] == "AV" :
    my_onshell_simple = xsec_monojet_axial(onshell_mMed, onshell_mDM, this_scenario["gq"], this_scenario["gDM"], this_scenario["gl"])
  else :
    my_onshell_simple = xsec_monojet_vector(onshell_mMed, onshell_mDM, this_scenario["gq"], this_scenario["gDM"], this_scenario["gl"])
  print("Simple on-shell scale factor:",paper_onshell_simple/my_onshell_simple)
  paper_onshell_medium = (paper_scenario["gq"]**2 * paper_scenario["gDM"]**2)/totalWidthAxial(onshell_mMed, onshell_mDM, paper_scenario["gq"], paper_scenario["gDM"], paper_scenario["gl"])
  print("\t",totalWidthAxial(onshell_mMed, onshell_mDM, paper_scenario["gq"], paper_scenario["gDM"], paper_scenario["gl"]),paper_scenario["gq"]**2, paper_scenario["gDM"]**2)
  if this_scenario["model"] == "AV" :
    my_onshell_medium = (this_scenario["gq"]**2 * this_scenario["gDM"]**2)/totalWidthAxial(onshell_mMed, onshell_mDM, this_scenario["gq"], this_scenario["gDM"], this_scenario["gl"])
    print("\t",totalWidthAxial(onshell_mMed, onshell_mDM, this_scenario["gq"], this_scenario["gDM"], this_scenario["gl"]),this_scenario["gq"]**2, this_scenario["gDM"]**2)
  else :
    my_onshell_medium = (this_scenario["gq"]**2 * this_scenario["gDM"]**2)/totalWidthVector(onshell_mMed, onshell_mDM, this_scenario["gq"], this_scenario["gDM"], this_scenario["gl"])
    print("\t",totalWidthVector(onshell_mMed, onshell_mDM, this_scenario["gq"], this_scenario["gDM"], this_scenario["gl"]),this_scenario["gq"]**2,this_scenario["gDM"]**2)
  print("Intermediate on-shell scale factor:",paper_onshell_medium/my_onshell_medium)
  paper_onshell_complex = relative_monojet_xsec_fixedmasses_axial(onshell_mMed, onshell_mDM, paper_scenario["gq"], paper_scenario["gDM"], paper_scenario["gl"])
  if this_scenario["model"] == "AV" :
    my_onshell_complex = relative_monojet_xsec_fixedmasses_axial(onshell_mMed, onshell_mDM, this_scenario["gq"], this_scenario["gDM"], this_scenario["gl"])
  else :
    my_onshell_complex = relative_monojet_xsec_fixedmasses_vector(onshell_mMed, onshell_mDM, this_scenario["gq"], this_scenario["gDM"], this_scenario["gl"])
  print("Complex on-shell scale factor:",paper_onshell_complex/my_onshell_complex)
  # paper_offshell_simple = paper_scenario["gq"]**2 * paper_scenario["gDM"]**2
  # my_offshell_simple = this_scenario["gq"]**2 * this_scenario["gDM"]**2
  # print("Simple off-shell scale factor:",paper_offshell_simple/my_offshell_simple)
  # paper_offshell_complex = relative_monojet_xsec_fixedmasses_axial(offshell_mMed, offshell_mDM, paper_scenario["gq"], paper_scenario["gDM"], paper_scenario["gl"])
  # if this_scenario["model"] == "AV" :
  #   my_offshell_complex = relative_monojet_xsec_fixedmasses_axial(offshell_mMed, offshell_mDM, this_scenario["gq"], this_scenario["gDM"], this_scenario["gl"])
  # else :
  #   my_offshell_complex = relative_monojet_xsec_fixedmasses_vector(offshell_mMed, offshell_mDM, this_scenario["gq"], this_scenario["gDM"], this_scenario["gl"])  
  # print("Complex off-shell scale factor:",paper_offshell_complex/my_offshell_complex)
  ## End of validation

  # paper_val = sigma_obs/sigma_theory
  # Want to convert to sigma_obs/sigma_new_theory
  # So multiply by (sigma_theory/sigma_new_theory)
  for (mMed, mDM, paper_val) in zip(xlist,ylist,zlist) :

    paper_xsec = relative_monojet_xsec_fixedmasses_axial(mMed, mDM, paper_scenario["gq"], paper_scenario["gDM"], paper_scenario["gl"])
    if this_scenario["model"] == "AV" :
      this_xsec = relative_monojet_xsec_fixedmasses_axial(mMed, mDM, this_scenario["gq"], this_scenario["gDM"], this_scenario["gl"])
    else :
      this_xsec = relative_monojet_xsec_fixedmasses_vector(mMed, mDM, this_scenario["gq"], this_scenario["gDM"], this_scenario["gl"])

    # We get zero cross section off-shell with this setup
    if this_xsec == 0 :
      continue
    else :
      converted_val = paper_val*(paper_xsec/this_xsec)
      converted_z.append(converted_val)
      converted_x.append(mMed)
      converted_y.append(mDM)

  # Now ready to make a plot.
  array_x = np.array(converted_x)
  array_y = np.array(converted_y)
  array_z = np.array(converted_z)
  make_plot(array_x,array_y,array_z,scenario)

  # and save.
  scenarios[scenario]["xvals"] = array_x
  scenarios[scenario]["yvals"] = array_y
  scenarios[scenario]["zvals"] = array_z


# Compare each of my contours to official ones.
import ROOT
inputDir = "/afs/cern.ch/work/k/kpachal/DMSP/dmcombinationandinterpretation/inputs/av_results_monojet_36ifb/"
scenario_files = {
  "AV_gq0p25_gchi1p0" : "Monojet_A1_MassMassplot_36ifb.root",
  "AV_gq0p1_gl0p1_gchi1p0" : "Monojet_A2_MassMassplot_36ifb.root",
  "V_gq0p25_gchi1p0" : "Monojet_V1_MassMassplot_36ifb.root",
  "V_gq0p1_gl0p01_gchi1p0" : "Monojet_V2_MassMassplot_36ifb.root"
}

from matplotlib.path import Path
for scenario in scenario_files.keys() :

  # Get TGraph and convert to arrays
  infile = ROOT.TFile.Open(inputDir+scenario_files[scenario],"READ")
  curve = infile.Get("obs_contour")
  xvals = []
  yvals = []
  for i in range(curve.GetN()) :
    xvals.append(curve.GetPointX(i))
    yvals.append(curve.GetPointY(i))

  # Now make a curve
  vertices = np.column_stack((xvals, yvals))
  # Doesn't seem to be working?
  treatments = [Path.MOVETO]+[Path.LINETO]*(vertices.size-1)

  string_path = mpath.Path(vertices)#, treatments)
  patch = mpatches.PathPatch(string_path, edgecolor="red", facecolor=None, fill=False, lw=2)

  make_plot(scenarios[scenario]["xvals"], scenarios[scenario]["yvals"], scenarios[scenario]["zvals"], scenario+"_compare", addCurves=[patch])


# and save in usable format? TBD.