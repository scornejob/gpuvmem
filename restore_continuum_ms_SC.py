#casapy --log2term --nogui -c restore_continum_ms_SC.py
print sys.argv
residual_ms = str(sys.argv[4]); #input
model_fits = str(sys.argv[5]);  # input
restored = str(sys.argv[6]);  #output
weight=str(sys.argv[7]); # "briggs or natural"
polarization="I"

######################################################################
residual_image=residual_ms+".img"

os.system("rm -rf *.log *.last "+residual_image+".* mod_out convolved_mod_out convolved_mod_out.fits "+restored+" "+restored+".fits")

importfits(imagename="mod_out",fitsimage=model_fits)

shape = imhead(imagename="mod_out", mode="get", hdkey="shape")
pix_num = shape[0]
cdelt = imhead(imagename="mod_out", mode="get", hdkey="cdelt2")
cdelta = qa.convert(v=cdelt,outunit="arcsec")
cdeltd = qa.convert(v=cdelt,outunit="deg")
pix_size = str(cdelta['value'])+"arcsec"
#print "pix_size",pix_size
#print "pix_num",pix_num



clean(vis=residual_ms, imagename=residual_image, mode='mfs', niter=0, stokes=polarization, weighting=weight, imsize=[pix_num,pix_num], cell=pix_size)

exportfits(imagename=residual_image+".image", fitsimage=residual_image+".image.fits")

ia.open(infile=residual_image+".image")
rbeam=ia.restoringbeam()
ia.done()
ia.close()

bmaj = imhead(imagename=residual_image+".image", mode="get", hdkey="beammajor")
bmin = imhead(imagename=residual_image+".image", mode="get", hdkey="beamminor")
bpa  = imhead(imagename=residual_image+".image", mode="get", hdkey="beampa")
#print "bmaj ",bmaj

#major = qa.convert(v=bmaj,outunit="deg")
#print "major ",major
#print "major value ",major['value']

minor = qa.convert(v=bmin,outunit="deg")
pa    = qa.convert(v=bpa ,outunit="deg")

#print "cdeltd", cdeltd
#print "log",log(2)

#DO NOT DELETE convert_factor = (pi/(4*log(2))) * major['value']* minor['value'] /  (cdeltd['value']**2)

#print "convert_factor",convert_factor,"\n"


ia.open(infile="mod_out")
ia.convolve2d(outfile="convolved_mod_out", axes=[0,1], type='gauss', major=bmaj, minor=bmin, pa=bpa)
ia.done()
ia.close()

exportfits(imagename="convolved_mod_out", fitsimage="convolved_mod_out.fits")
ia.open(infile="convolved_mod_out.fits")
ia.setrestoringbeam(beam=rbeam)
ia.done()
ia.close()

imagearr=["convolved_mod_out.fits",residual_image+".image.fits"]

#immath(imagename=imagearr,expr=" (IM0 * convert_factor  + IM1) ", outfile=restored)

immath(imagename=imagearr,expr=" (IM0   + IM1) ", outfile=restored)

exportfits(imagename=restored, fitsimage=restored+".fits",overwrite=True)
