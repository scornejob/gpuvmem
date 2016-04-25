#!/usr/bin/perl -w
use PDL;
use Data::Dumper;
####pastes beam info from all pointings into the beam_*.fits files, inplace.

#$infile = $ARGV[0];
#
#$in  = rfits($infile);
#$h = $in -> gethdr;

opendir DIR,'.';
@files = readdir DIR;
foreach $name  (@files) {
    $name =~ /beam_(.+)\.fits/;
    $i = $1;
    if ((defined($i)) &&  ($i =~ /\d/)) {
	$in = "mod_in_$i.fits";
	print "$name ---> $in \n";
	$beam = rfits($name);
	$hbeam = $beam -> gethdr;

	$im = rfits($in);
	$h = $im->gethdr;
	
#	print Dumper($h);
#	print Dumper($hbeam);


	$$hbeam{'BMAJ'} = $$h{'BMAJ'}/abs($$h{'CDELT2'});
	$$hbeam{'BMIN'} = $$h{'BMIN'}/abs($$h{'CDELT2'});
	$$hbeam{'BPA'} = $$h{'BPA'};

	$beam -> sethdr($hbeam);
	wfits $beam,$name;
    }
}
