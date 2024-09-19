import pylab as pl
import numpy as np
data = np.loadtxt('wannier90-bands.dat')
x=data[:,0]
y=data[:,1]
z=data[:,2]
tick_labels=[]
tick_locs=[]
tick_labels.append(' K'.strip())
tick_locs.append(0)
tick_labels.append('$\Gamma$')
tick_locs.append(    1.222001)
tick_labels.append(' M'.strip())
tick_locs.append(    2.280287)
tick_labels.append(' K'.strip())
tick_locs.append(    2.891287)
pl.scatter(x,y,c=z,marker='+',s=1,cmap=pl.cm.jet)
pl.xlim([0,max(x)])
pl.ylim([min(y)-0.025*(max(y)-min(y)),max(y)+0.025*(max(y)-min(y))])
pl.xticks(tick_locs,tick_labels)
for n in range(1,len(tick_locs)):
   pl.plot([tick_locs[n],tick_locs[n]],[pl.ylim()[0],pl.ylim()[1]],color='gray',linestyle='-',linewidth=0.5)
pl.ylabel('Energy [eV]')
pl.axes().set_aspect(aspect=0.65*max(x)/(max(y)-min(y)))
pl.colorbar(shrink=0.7)
outfile = 'wannier90-bands.pdf'
pl.savefig(outfile,bbox_inches='tight')
pl.show()
