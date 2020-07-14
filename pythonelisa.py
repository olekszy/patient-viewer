import sys
import pandas as pd
import glob,os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.interpolate import interp1d
from pick import pick
from tabulate import tabulate
#platetemplate = sys.argv[1] #Read template of plate
#csv = sys.argv[2] #Read csv file from reader
def import1d(x):
    df = pd.read_csv(x, sep = ';', index_col=0)
    out = np.empty(df.shape[0], dtype=object)
    out[:] = df.values.tolist()
    return out
def importdata(x):
    df = pd.read_csv(x, sep = ';', index_col=0)
    df = df.apply(lambda x: x.str.replace(',','.'))
    return df.astype("float")
def calcc(y):
    a = []
    const = (max(y)-min(y))/2
    for i in y :
        z = np.absolute(i-const)
        a.append(z)
    return min(a)

def calcC(y):
    return np.median(y)

def logistic4(x, A, B, C, D):
    """4PL lgoistic equation."""
    return ((A-D)/(1.0+((C/x)**B)))+D # Logistic equation

def residuals(p, y, x): # Deviations of Data
    """Deviations of data from fitted 4PL curve"""
    A,B,C,D = p
    err = y-logistic4(x, A, B, C, D) 
    return err

def peval(x, p):
    """Evaluated value at x with current parameters."""
    A,B,C,D = p
    return logistic4(x, A, B, C, D)

def concentration(y,D,B,C,A): #exchanged D and A paramters in scipy
    x = C*(((A-D)/(y-D))-1)**(1/B)
    return x  

def cuttable(x, wave):
    z = pd.read_csv(x, sep=";",header = None).fillna("")
    row = z[z[1].str.contains(wave)].index.values.astype(int)[0]
    name = z.iloc[row,1]
    table = z.iloc[row+2:row+10]
    table = pd.DataFrame(table).set_index([0])
    table = table.drop(columns=table.columns[(table == '').any()])
    table = table.apply(lambda x: x.str.replace(',','.'))
    return table.astype("float")
def r2value(y, y_predicted):
    ss_res = np.sum((y - y_predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2
def readsample(temp,df,S):
    z = []
    #Read Standards 
    for column in temp.columns: # search  in columns
        for row in temp.index: #search in rows
            y = temp.at[row,column] #find specific place
            if S in y:
                z.append(row+":"+column) #fill list with locations of samples
    value = [] #create list for extracted values
    for i in z: 
        x = i[0] # take rows
        y = int(i[2:]) # take column
        value.append(df.at[x,y]) #fill list with extracted values of samples
        print (value) 
        avgvalue = np.mean(value) #count averages of repetitions
    return value, avgvalue.item()
def xsamples(temp,df,popt):
    results = pd.DataFrame({"Sample":[], "Concentration":[], "QC":[]}) #create new dataframe
    samplelist = np.unique(temp.iloc[:].values) #extract sample names
    for i in samplelist: #loop over sample names extracted before
        s = 1
        a,b = readsample(temp, df, i) #Read samples from template
        lenght = len(a)
        mean = np.mean(a) #count mean value from repetitions
        #part with std errors
        threshold = mean*0.25 #threshold assigned as 25%
        diffplus = mean + threshold 
        diffminus = mean - threshold
        for m in a: #for loop to check if value is in 25% range 
            if not (diffminus <= m <= diffplus):
                qc = "SD" #QC not passed
                break
            else:
                qc = "PASSED"
        #appending pd
        z = concentration(b, *popt.tolist()) #Calculate concentration from 4PL Curve
        temporary = pd.DataFrame({"Sample":[i], "Concentration":[z.real],"QC":[qc]}) #create temporary table for better appending
        for reads in a: #append reads to csv in appropriate columns
            while s > lenght:
                break
            temporary["read"+str(s)] = round(reads, 3) #round results to three places after ,
            print("added" + "read"+str(s))
            s = s+1
        #print(temporary) #Check
        results = results.append(temporary, ignore_index=True) #append results to table
        #print(z) #Check
    return results
def ODbyELISA(results):
    z = results[results['Sample'].str.contains("1Standard")].values # extract 1Standard to 1000
    const = z.flat[1] #extract value
    conc = results['Concentration'].tolist() #change column to list
    lista = [] #create list for append
    #print(const)
    for i in conc:    
        x = (1000*i)/const #formula over all rows
        lista.append(x)
    results["ELISA units"] = lista #create column from list
    return results
def omitstandards(standards,measured,diluted,name):
    title = 'Please choose Standards to omit in ' + name
    options = standards
    selected = pick(options, title, multiselect=True, min_selection_count=0)
    try:
        print(selected)
        f = standards.index(selected[0][0])
        standards.remove(selected[0][0])
        y = np.delete(measured,f) #Delete measrure
        x = np.delete(diluted,f) #Delete diluted
        print ("Omiting " + str(selected[0][0]))
        return f, y, x 
    except IndexError:
        return standards, measured, diluted
def createimage(x,y,xnew,f,f2,r2valuestr):
        fig = plt.figure()
        plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
        plt.title(csv+" "+"4PL curve", fontdict=None, loc='center', pad=None)
        plt.legend(['data', 'linear', 'cubic'], loc='best')
        plt.xlabel("Concentration")
        plt.ylabel("OD")
        #plt.text(2, 0.65,"R2 value " + r2valuestr , fontdict=font)
        return fig

################################## PROGRAM STARTS HERE #################################################
def analysis(csv,platetemplate, standardsnumber,value,dil,filename): 
    template = pd.read_csv(platetemplate, sep = ";", index_col=0) # Read sample sheets  
    print(csv)
    df450 = cuttable(csv, "(450)") #Cut 450 nm table
    df570 = cuttable(csv, "(570)") #Cut 570 nm table
    print("Properly imported plates")
    #Background substraction 
    dfavg  = df450.sub(df570) #Cut background
    print("Plates substracted")
    #Read Standards 
    standards = [] # Create standards
    #countstandards = int(input("How many standards did you use ")) #input number of standards

    for i in range(1,standardsnumber+1):
        x = str(i)+"Standard"
        standards.append(x)# append list of S1..S2.. etc
    np.unique(standards.append("B"))

    ##### Define Concentrations
    #value = input("Starting Concentration ") #Input number of standards
    #Input Dilution factor
    #dil = float(input("Dilution factor ")) #Change to float

    list = [] # create list
    list.append(value) #Add starting value

    for i in range(countstandards-1):
        value = float(value)/dil #Create series for standarization
        list.append(value) # append series values
    list.append(0)
    a = np.asarray(list) # save as array
    a = np.loadtxt(a, dtype='float') #delete dtype at the end
    x = a
    print("Standards created")
    col = []

    for i in standards:   
        z,avg = readsample(template, dfavg, i)
        col.append(avg)
    col = np.asarray(col)
    #Attach Standards OD
    y = col

    x=a #crucial!!!!!!!
    print("Standards attached")
    #print(x)

    null, y, x = omitstandards(standards,col,a, filename)

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(logistic4, x, y)
    y_true = logistic4(x, *popt)

    f = interp1d(x, y_true)
    f2 = interp1d(x, y_true, kind='cubic')

    xnew = np.linspace(min(x), max(x))

    print("ABCD paramteres completed")

    # Create plot and save it 
    #Firstly, check if results folder exists
    import os
    if os.path.isdir("results") == False:
        os.makedirs('results')
    r = r2value(y, y_true)
    r2valuestr = str(r)
    
    z = createimage(x, y, xnew,f,f2,r2valuestr)
    
    print("Image created")
    
    print("R2 value " + r2valuestr)

    print("Saving results to files")
    filename = csv.split("/")[-1]
    print(filename[0:-4])

    z.savefig("results/"+filename[0:-4])

    results = xsamples(template,dfavg,popt)
    final = ODbyELISA(results)

    return final, r

from os import listdir
from os.path import isfile, join

workdir = "workfolder"
sheets = "samplesheets"
samples = [f for f in listdir(workdir) if isfile(join(workdir, f))] #find folders for samples

templates = [f for f in listdir(sheets) if isfile(join(sheets, f))] #find folders for templates
title = 'Please choose files to analyse'

#option, index = pick(samples, title)
chosensamples = pick(samples, title, multiselect=True, min_selection_count=0)

print (chosensamples)

samplelist = []
for samples in chosensamples:
    title = "Attach template to " + str(samples[0])
    sheet, index = pick(templates, title)
    print(sheet +" attached to " + str(samples[0]))
    samplelist.append(samples[0]+":"+sheet)

finalresults = pd.DataFrame()
standards = [] # Create standards
countstandards = int(input("How many standards did you use ")) #input number of standards

##### Define Concentrations
value = input("Starting Concentration ") #Input number of standards
#Input Dilution factor
dil = float(input("Dilution factor ")) #Change to float

for sample in samplelist:
    csv,platetemplate = sample.split(":")
    csv = workdir+"/"+csv
    platetemplate = sheets+"/"+platetemplate
    fn = str(csv) 
    print("Start analysis" + csv + " with template " + platetemplate)
    final, r2 = analysis(csv,platetemplate,countstandards,value,dil,fn)
    final["R^2"] = round(r2,4)
    print(tabulate(final, headers='keys', tablefmt='psql'))
    finalresults = finalresults.append(final)

name = input("How to save your analysis?")
finalresults = finalresults[~finalresults['Sample'].isin(["B", "no antigen"])]
#print(finalresults)
finalresults.to_csv("results/"+name+".csv", sep = "\t")
