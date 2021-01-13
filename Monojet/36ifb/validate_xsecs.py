import numpy as np
import matplotlib.pyplot as plt
from monojet_functions import *
import scipy
import json
import lhapdf

# test test

gq = 0.25
gl = 0
gDM = 0.1

# Pick some values where I see disagreement in the plot.
with open("original_limits_AV_gq0p25_gchi1p0.json", "r") as read_file:
  data = json.load(read_file)["0.25"]

# Let's understand what's going on around 1000, 500ish:
test_points = [i for i in data if (i["x"]=="1000" and eval(i["y"])>400 and eval(i["y"])<600)]

def manual_integration(xlist,Gamma,isVec) :
  ylist = []
  for x in xlist :
    if isVec :    
      val = integrand_vector(x,Gamma,M,mDM)
    else :
      val = integrand_axialvector(x,Gamma,M,mDM)
    if np.isnan(val) :
      print("Got a nan! Inputs were: ", x,Gamma,M,mDM,"and isVec is",isVec)
    ylist.append(val)
  thisresult = scipy.integrate.trapz(ylist,x=xlist)
  return thisresult

for point in sorted(test_points,key=lambda k: k["y"]) :

  M = eval(point["x"])
  mDM = eval(point["y"])
  original_limit = eval(point["value"])
  tag = "testpoint_M{0}_mDM{1}".format(point["x"],point["y"])
  print("\nBeginning",tag)

  # Going to plot S values
  # Hypothesis: these points also not good enough when we're very close to off-shell.
  # Close to off-shell, peak is to the left of where we actually start.  
  lowVal = 4*mDM**2
  peakVal = M**2
  endVal = 169000000
  #endVal = 3e6
  Slist = np.logspace(np.log10(lowVal), np.log10(endVal), 20000, endpoint=True)
  #Slist = np.rint(Slist)
  # Slist = [lowVal, lowVal+1,lowVal+2]
  # Slist = Slist+list(range(lowVal+2,peakVal,100))
  # Slist = Slist+list(range(max(lowVal+2,peakVal),2*peakVal,100))
  # Slist = Slist+list(range(2*peakVal,169000000,10000))
  # Slist.append(169000000)
  S = np.rint(np.array(Slist))
  print(S)
  print("Manually evaluating at",len(Slist),"points.")


  plt.clf()

  xsecs = {}

  # Do vector and axial vector
  for model in ["vector","axialvector"] :

    print("Begin testing",model,"model.")

    if model == "vector": isVector = True
    else : isVector = False

    if isVector :
      Gamma = totalWidthVector(M, mDM, gq, gDM, gl)
      Y = np.array([integrand_vector(i,Gamma,M,mDM) for i in S])    
    else :
      Gamma = totalWidthAxial(M, mDM, gq, gDM, gl)    
      Y = np.array([integrand_axialvector(i,Gamma,M,mDM) for i in S])

    # Create the plot
    plt.plot(S,Y)
    plt.yscale("log")
    if isVector :
      plt.ylim(Y[-1], 5*np.amax(Y))
    plt.xscale("log")  

    # Show the plot (only interesting for overlay, i.e. second iteration)
    if not isVector :
      plt.savefig('plots/plot_dsigdS_{0}.eps'.format(tag))

    # Now do the integral.
    intpoints = [M,M**2-Gamma,M**2,M**2+Gamma]
    if isVector :
      integral = integrate.quad(integrand_vector,lowVal,endVal,args=(Gamma,M,mDM),points=intpoints,full_output=1)
    else :
      integral = integrate.quad(integrand_axialvector,lowVal,endVal,args=(Gamma,M,mDM),points=intpoints,full_output=1)
    print(integral[0])

    # Now test manually integrating at every step
    print("Calculating manual integral...")
    thisval = manual_integration(S,Gamma,isVector) # np.array(range(4,13000**2,4))
    print(thisval)

    # Do they agree? Yes.
    xsecs[model] = thisval

  # OK. What does that mean for exclusion?
  print("Original value:",original_limit)
  print("Original (axial vector) point",("was not excluded" if original_limit>1. else "was excluded."))
  ratio = xsecs["axialvector"]/xsecs["vector"]
  newval = original_limit*ratio
  print("New value:",newval)
  print("New (vector) point",("was not excluded" if newval>1. else "was excluded."))
  if(newval > 1) :
    print("Would need V to AV cross-section ratio to be",original_limit,"or larger to exclude point.")
    print("Ratio was only",1./ratio)

# Do some lhapdf testing.
lhapdf.pathsPrepend("/cvmfs/sft.cern.ch/lcg/releases/LCG_97python3/MCGenerators/lhapdf/6.2.3/x86_64-centos7-gcc9-opt/share/LHAPDF/")
lhapdf.pathsPrepend("/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current/")
#pdfs = [i for i in lhapdf.availablePDFSets() if "NNPDF30_lo" in i]
pdf = lhapdf.mkPDF("NNPDF30_lo_as_0118_nf_4",0)

print("Able to use it?")
# Test:
for pid in pdf.flavors():
  print("PID",pid)
  print(pdf.xfxQ(pid, 0.01, 91.2))

# 4 flavour scheme. u, d, s, c. Need u \bar(u), etc. So:

#p = lhapdf.mkPDF("CT10", 0)
#print(p.xfxQ2(21, 1e-3, 1e4))
