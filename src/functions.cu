/* -------------------------------------------------------------------------
  Copyright (C) 2016-2017  Miguel Carcamo, Pablo Roman, Simon Casassus,
  Victor Moral, Fernando Rannou - miguel.carcamo@usach.cl

  This program includes Numerical Recipes (NR) based routines whose
  copyright is held by the NR authors. If NR routines are included,
  you are required to comply with the licensing set forth there.

	Part of the program also relies on an an ANSI C library for multi-stream
	random number generation from the related Prentice-Hall textbook
	Discrete-Event Simulation: A First Course by Steve Park and Larry Leemis,
  for more information please contact leemis@math.wm.edu

  For the original parts of this code, the following license applies:

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
* -------------------------------------------------------------------------
*/
#include "functions.cuh"


extern long M, N;
extern int numVisibilities, iterations, iterthreadsVectorNN, blocksVectorNN, nopositivity, crpix1, crpix2, \
status_mod_in, verbose_flag, apply_noise, clip_flag, num_gpus, selected, iter, t_telescope, multigpu, firstgpu, reg_term, print_images;

extern cufftHandle plan1GPU;
extern cufftComplex *device_V, *device_Inu;

extern float2 *device_dchi2_total, *device_2I;
extern float *device_chi2, *device_S, *device_dS, *device_noise_image;
extern float noise_jypix, fg_scale, DELTAX, DELTAY, deltau, deltav, noise_cut, MINPIX, \
minpix, lambda, ftol, random_probability, final_chi2, nu_0, final_H, alpha_start;

extern dim3 threadsPerBlockNN, numBlocksNN;

extern float beam_noise, beam_bmaj, beam_bmin, b_noise_aux, beam_fwhm, beam_freq, beam_cutoff, eta;
extern double ra, dec;

extern freqData data;

extern char* mempath, *out_image;

extern fitsfile *mod_in;

extern Field *fields;

extern int flag_opt;

extern VariablesPerField *vars_per_field;

__host__ void goToError()
{
  if(num_gpus > 1){
    for(int i=firstgpu+1; i<firstgpu + num_gpus; i++){
          cudaSetDevice(firstgpu);
          cudaDeviceDisablePeerAccess(i);
          cudaSetDevice(i);
          cudaDeviceDisablePeerAccess(firstgpu);
    }

    for(int i=0; i<num_gpus; i++ ){
          cudaSetDevice((i%num_gpus) + firstgpu);
          cudaDeviceReset();
    }
  }

  printf("An error has ocurred, exiting\n");
  exit(0);

}

__host__ void init_beam(int telescope)
{
  switch(telescope) {
  case 1:
    beam_fwhm = 33.0*RPARCM;   /* radians CBI2 */
    beam_freq = 30.0;          /* GHz */
    beam_cutoff = 90.0*RPARCM; /* radians */
    break;
  case 2:
    beam_fwhm = (8.4220/60)*RPARCM;   /* radians ALMA */
    beam_freq = 691.4;          /* GHz */
    beam_cutoff = 1.0*RPARCM; /* radians */
    break;
  case 3: //test
    beam_fwhm = 5*RPARCM;   /* radians CBI2 */
    beam_freq = 1000;          /* GHz */
    beam_cutoff = 10*RPARCM; /* radians */
    break;
  case 4:
    beam_fwhm = (9.0/60)*RPARCM*12/22;   /* radians ATCA */
    beam_freq = 691.4;          /* GHz */
    beam_cutoff = 1.0*RPARCM; /* radians */
    break;
  case 5:
    beam_fwhm = (9.0/60)*RPARCM*12/25;   /* radians VLA */
    beam_freq = 691.4;          /* GHz */
    beam_cutoff = 1.0*RPARCM; /* radians */
    break;
  case 6:
    beam_fwhm = 10.5*RPARCM;   /* radians SZA */
    beam_freq = 30.9380;          /* GHz */
    beam_cutoff = 20.0*RPARCM; /* radians */
    break;

  default:
    printf("Telescope type not defined\n");
    goToError();
    break;
  }
}


__host__ long NearestPowerOf2(long x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}


__host__ void readInputDat(char *file)
{
  FILE *fp;
  char item[50];
  float status;
  if((fp = fopen(file, "r")) == NULL){
    printf("ERROR. The input file wasn't provided by the user.\n");
    goToError();
  }else{
    while(true){
      int ret = fscanf(fp, "%s %e", item, &status);

      if(ret==EOF){
        break;
      }else{
        if (strcmp(item,"lambda_entropy")==0) {
          if(lambda == -1){
            lambda = status;
          }
        }else if (strcmp(item,"noise_cut")==0){
          if(noise_cut == -1){
            noise_cut = status;
          }
        }else if (strcmp(item,"t_telescope")==0){
          t_telescope = status;
        }else if(strcmp(item,"minpix")==0){
          if(minpix == -1){
            minpix = status;
          }
        } else if(strcmp(item,"ftol")==0){
          ftol = status;
        } else if(strcmp(item,"random_probability")==0){
          if(random_probability == -1){
            random_probability = status;
          }
        }else{
          printf("Keyword not defined in input\n");
          goToError();
        }
      }
    }
  }
}

__host__ void print_warranty() {
  printf("THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY \
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT \
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM 'AS IS' WITHOUT WARRANTY \
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, \
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR \
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM \
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF \
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.\n");
}

__host__ void print_copyright() {
  printf("   TERMS AND CONDITIONS \n"
  " \n"
  "0. Definitions. \n"
  " \n"
  "'This License' refers to version 3 of the GNU General Public License. \n"
  " \n"
  "'Copyright' also means copyright-like laws that apply to other kinds of \n"
  "works, such as semiconductor masks. \n"
  " \n"
  "'The Program' refers to any copyrightable work licensed under this \n"
  "License.  Each licensee is addressed as 'you'.  'Licensees' and \n"
  "'recipients' may be individuals or organizations. \n"
  " \n"
  "To 'modify' a work means to copy from or adapt all or part of the work \n"
  "in a fashion requiring copyright permission, other than the making of an \n"
  "exact copy.  The resulting work is called a 'modified version' of the \n"
  "earlier work or a work 'based on' the earlier work. \n"
  " \n"
  "A 'covered work' means either the unmodified Program or a work based \n"
  "on the Program. \n"
  " \n"
  "To 'propagate' a work means to do anything with it that, without \n"
  "permission, would make you directly or secondarily liable for \n"
  "infringement under applicable copyright law, except executing it on a \n"
  "computer or modifying a private copy.  Propagation includes copying, \n"
  "distribution (with or without modification), making available to the \n"
  "public, and in some countries other activities as well. \n"
  " \n"
  "To 'convey' a work means any kind of propagation that enables other \n"
  "parties to make or receive copies.  Mere interaction with a user through \n"
  "a computer network, with no transfer of a copy, is not conveying. \n"
  " \n"
  "An interactive user interface displays 'Appropriate Legal Notices' \n"
  "to the extent that it includes a convenient and prominently visible \n"
  "feature that (1) displays an appropriate copyright notice, and (2) \n"
  "tells the user that there is no warranty for the work (except to the \n"
  "extent that warranties are provided), that licensees may convey the \n"
  "work under this License, and how to view a copy of this License.  If \n"
  "the interface presents a list of user commands or options, such as a \n"
  "menu, a prominent item in the list meets this criterion. \n"
  " \n"
  "1. Source Code. \n"
  " \n"
  "The 'source code' for a work means the preferred form of the work \n"
  "for making modifications to it.  'Object code' means any non-source \n"
  "form of a work. \n"
  " \n"
  "A 'Standard Interface' means an interface that either is an official \n"
  "standard defined by a recognized standards body, or, in the case of \n"
  "interfaces specified for a particular programming language, one that \n"
  "is widely used among developers working in that language. \n"
  " \n"
  "The 'System Libraries' of an executable work include anything, other \n"
  "than the work as a whole, that (a) is included in the normal form of \n"
  "packaging a Major Component, but which is not part of that Major \n"
  "Component, and (b) serves only to enable use of the work with that \n"
  "Major Component, or to implement a Standard Interface for which an \n"
  "implementation is available to the public in source code form.  A \n"
  "'Major Component', in this context, means a major essential component \n"
  "(kernel, window system, and so on) of the specific operating system \n"
  "(if any) on which the executable work runs, or a compiler used to \n"
  "produce the work, or an object code interpreter used to run it. \n"
  " \n"
  "The 'Corresponding Source' for a work in object code form means all \n"
  "the source code needed to generate, install, and (for an executable \n"
  "work) run the object code and to modify the work, including scripts to \n"
  "control those activities.  However, it does not include the work's \n"
  "System Libraries, or general-purpose tools or generally available free \n"
  "programs which are used unmodified in performing those activities but \n"
  "which are not part of the work.  For example, Corresponding Source \n"
  "includes interface definition files associated with source files for \n"
  "the work, and the source code for shared libraries and dynamically \n"
  "linked subprograms that the work is specifically designed to require, \n"
  "such as by intimate data communication or control flow between those \n"
  "subprograms and other parts of the work. \n"
  " \n"
  "The Corresponding Source need not include anything that users \n"
  "can regenerate automatically from other parts of the Corresponding \n"
  "Source. \n"
  " \n"
  "The Corresponding Source for a work in source code form is that \n"
  "same work. \n"
  " \n"
  "2. Basic Permissions. \n"
  " \n"
  "All rights granted under this License are granted for the term of \n"
  "copyright on the Program, and are irrevocable provided the stated \n"
  "conditions are met.  This License explicitly affirms your unlimited \n"
  "permission to run the unmodified Program.  The output from running a \n"
  "covered work is covered by this License only if the output, given its \n"
  "content, constitutes a covered work.  This License acknowledges your \n"
  "rights of fair use or other equivalent, as provided by copyright law. \n"
  " \n"
  "You may make, run and propagate covered works that you do not \n"
  "convey, without conditions so long as your license otherwise remains \n"
  "in force.  You may convey covered works to others for the sole purpose \n"
  "of having them make modifications exclusively for you, or provide you \n"
  "with facilities for running those works, provided that you comply with \n"
  "the terms of this License in conveying all material for which you do \n"
  "not control copyright.  Those thus making or running the covered works \n"
  "for you must do so exclusively on your behalf, under your direction \n"
  "and control, on terms that prohibit them from making any copies of \n"
  "your copyrighted material outside their relationship with you. \n"
  " \n"
  "Conveying under any other circumstances is permitted solely under \n"
  "the conditions stated below.  Sublicensing is not allowed; section 10 \n"
  "makes it unnecessary. \n"
  " \n"
  "3. Protecting Users' Legal Rights From Anti-Circumvention Law. \n"
  " \n"
  "No covered work shall be deemed part of an effective technological \n"
  "measure under any applicable law fulfilling obligations under article \n"
  "11 of the WIPO copyright treaty adopted on 20 December 1996, or \n"
  "similar laws prohibiting or restricting circumvention of such \n"
  "measures. \n"
  " \n"
  "When you convey a covered work, you waive any legal power to forbid \n"
  "circumvention of technological measures to the extent such circumvention \n"
  "is effected by exercising rights under this License with respect to \n"
  "the covered work, and you disclaim any intention to limit operation or \n"
  "modification of the work as a means of enforcing, against the work's \n"
  "users, your or third parties' legal rights to forbid circumvention of \n"
  "technological measures. \n"
  " \n"
  "4. Conveying Verbatim Copies. \n"
  " \n"
  "You may convey verbatim copies of the Program's source code as you \n"
  "receive it, in any medium, provided that you conspicuously and \n"
  "appropriately publish on each copy an appropriate copyright notice; \n"
  "keep intact all notices stating that this License and any \n"
  "non-permissive terms added in accord with section 7 apply to the code; \n"
  "keep intact all notices of the absence of any warranty; and give all \n"
  "recipients a copy of this License along with the Program. \n"
  " \n"
  "You may charge any price or no price for each copy that you convey, \n"
  "and you may offer support or warranty protection for a fee. \n"
  " \n"
  "5. Conveying Modified Source Versions. \n"
  " \n"
  "You may convey a work based on the Program, or the modifications to \n"
  "produce it from the Program, in the form of source code under the \n"
  "terms of section 4, provided that you also meet all of these conditions: \n"
  " \n"
  "a) The work must carry prominent notices stating that you modified \n"
  "it, and giving a relevant date. \n"
  " \n"
  "b) The work must carry prominent notices stating that it is \n"
  "released under this License and any conditions added under section \n"
  "7.  This requirement modifies the requirement in section 4 to \n"
  "'keep intact all notices'. \n"
  " \n"
  "c) You must license the entire work, as a whole, under this \n"
  "License to anyone who comes into possession of a copy.  This \n"
  "License will therefore apply, along with any applicable section 7 \n"
  "additional terms, to the whole of the work, and all its parts, \n"
  "regardless of how they are packaged.  This License gives no \n"
  "permission to license the work in any other way, but it does not \n"
  "invalidate such permission if you have separately received it. \n"
  " \n"
  "d) If the work has interactive user interfaces, each must display \n"
  "Appropriate Legal Notices; however, if the Program has interactive \n"
  "interfaces that do not display Appropriate Legal Notices, your \n"
  "work need not make them do so. \n"
  " \n"
  "A compilation of a covered work with other separate and independent \n"
  "works, which are not by their nature extensions of the covered work, \n"
  "and which are not combined with it such as to form a larger program, \n"
  "in or on a volume of a storage or distribution medium, is called an \n"
  "'aggregate' if the compilation and its resulting copyright are not \n"
  "used to limit the access or legal rights of the compilation's users \n"
  "beyond what the individual works permit.  Inclusion of a covered work \n"
  "in an aggregate does not cause this License to apply to the other \n"
  "parts of the aggregate. \n"
  " \n"
  "6. Conveying Non-Source Forms. \n"
  " \n"
  "You may convey a covered work in object code form under the terms \n"
  "of sections 4 and 5, provided that you also convey the \n"
  "machine-readable Corresponding Source under the terms of this License, \n"
  "in one of these ways: \n"
  " \n"
  "a) Convey the object code in, or embodied in, a physical product \n"
  "(including a physical distribution medium), accompanied by the \n"
  "Corresponding Source fixed on a durable physical medium \n"
  "customarily used for software interchange. \n"
  " \n"
  "b) Convey the object code in, or embodied in, a physical product \n"
  "(including a physical distribution medium), accompanied by a \n"
  "written offer, valid for at least three years and valid for as \n"
  "long as you offer spare parts or customer support for that product \n"
  "model, to give anyone who possesses the object code either (1) a \n"
  "copy of the Corresponding Source for all the software in the \n"
  "product that is covered by this License, on a durable physical \n"
  "medium customarily used for software interchange, for a price no \n"
  "more than your reasonable cost of physically performing this \n"
  "conveying of source, or (2) access to copy the \n"
  "Corresponding Source from a network server at no charge. \n"
  " \n"
  "c) Convey individual copies of the object code with a copy of the \n"
  "written offer to provide the Corresponding Source.  This \n"
  "alternative is allowed only occasionally and noncommercially, and \n"
  "only if you received the object code with such an offer, in accord \n"
  "with subsection 6b. \n"
  " \n"
  "d) Convey the object code by offering access from a designated \n"
  "place (gratis or for a charge), and offer equivalent access to the \n"
  "Corresponding Source in the same way through the same place at no \n"
  "further charge.  You need not require recipients to copy the \n"
  "Corresponding Source along with the object code.  If the place to \n"
  "copy the object code is a network server, the Corresponding Source \n"
  "may be on a different server (operated by you or a third party) \n"
  "that supports equivalent copying facilities, provided you maintain \n"
  "clear directions next to the object code saying where to find the \n"
  "Corresponding Source.  Regardless of what server hosts the \n"
  "Corresponding Source, you remain obligated to ensure that it is \n"
  "available for as long as needed to satisfy these requirements. \n"
  " \n"
  "e) Convey the object code using peer-to-peer transmission, provided \n"
  "you inform other peers where the object code and Corresponding \n"
  "Source of the work are being offered to the general public at no \n"
  "charge under subsection 6d. \n"
  " \n"
  "A separable portion of the object code, whose source code is excluded \n"
  "from the Corresponding Source as a System Library, need not be \n"
  "included in conveying the object code work. \n"
  " \n"
  "A 'User Product' is either (1) a 'consumer product', which means any \n"
  "tangible personal property which is normally used for personal, family, \n"
  "or household purposes, or (2) anything designed or sold for incorporation \n"
  "into a dwelling.  In determining whether a product is a consumer product, \n"
  "doubtful cases shall be resolved in favor of coverage.  For a particular \n"
  "product received by a particular user, 'normally used' refers to a \n"
  "typical or common use of that class of product, regardless of the status \n"
  "of the particular user or of the way in which the particular user \n"
  "actually uses, or expects or is expected to use, the product.  A product \n"
  "is a consumer product regardless of whether the product has substantial \n"
  "commercial, industrial or non-consumer uses, unless such uses represent \n"
  "the only significant mode of use of the product. \n"
  " \n"
  "'Installation Information' for a User Product means any methods, \n"
  "procedures, authorization keys, or other information required to install \n"
  "and execute modified versions of a covered work in that User Product from \n"
  "a modified version of its Corresponding Source.  The information must \n"
  "suffice to ensure that the continued functioning of the modified object \n"
  "code is in no case prevented or interfered with solely because \n"
  "modification has been made. \n"
  " \n"
  "If you convey an object code work under this section in, or with, or \n"
  "specifically for use in, a User Product, and the conveying occurs as \n"
  "part of a transaction in which the right of possession and use of the \n"
  "User Product is transferred to the recipient in perpetuity or for a \n"
  "fixed term (regardless of how the transaction is characterized), the \n"
  "Corresponding Source conveyed under this section must be accompanied \n"
  "by the Installation Information.  But this requirement does not apply \n"
  "if neither you nor any third party retains the ability to install \n"
  "modified object code on the User Product (for example, the work has \n"
  "been installed in ROM). \n"
  " \n"
  "The requirement to provide Installation Information does not include a \n"
  "requirement to continue to provide support service, warranty, or updates \n"
  "for a work that has been modified or installed by the recipient, or for \n"
  "the User Product in which it has been modified or installed.  Access to a \n"
  "network may be denied when the modification itself materially and \n"
  "adversely affects the operation of the network or violates the rules and \n"
  "protocols for communication across the network. \n"
  " \n"
  "Corresponding Source conveyed, and Installation Information provided, \n"
  "in accord with this section must be in a format that is publicly \n"
  "documented (and with an implementation available to the public in \n"
  "source code form), and must require no special password or key for \n"
  "unpacking, reading or copying. \n"
  " \n"
  "7. Additional Terms. \n"
  " \n"
  "'Additional permissions' are terms that supplement the terms of this \n"
  "License by making exceptions from one or more of its conditions. \n"
  "Additional permissions that are applicable to the entire Program shall \n"
  "be treated as though they were included in this License, to the extent \n"
  "that they are valid under applicable law.  If additional permissions \n"
  "apply only to part of the Program, that part may be used separately \n"
  "under those permissions, but the entire Program remains governed by \n"
  "this License without regard to the additional permissions. \n"
  " \n"
  "When you convey a copy of a covered work, you may at your option \n"
  "remove any additional permissions from that copy, or from any part of \n"
  "it.  (Additional permissions may be written to require their own \n"
  "removal in certain cases when you modify the work.)  You may place \n"
  "additional permissions on material, added by you to a covered work, \n"
  "for which you have or can give appropriate copyright permission. \n"
  " \n"
  "Notwithstanding any other provision of this License, for material you \n"
  "add to a covered work, you may (if authorized by the copyright holders of \n"
  "that material) supplement the terms of this License with terms: \n"
  " \n"
  "a) Disclaiming warranty or limiting liability differently from the \n"
  "terms of sections 15 and 16 of this License; or \n"
  " \n"
  "b) Requiring preservation of specified reasonable legal notices or \n"
  "author attributions in that material or in the Appropriate Legal \n"
  "Notices displayed by works containing it; or \n"
  " \n"
  "c) Prohibiting misrepresentation of the origin of that material, or \n"
  "requiring that modified versions of such material be marked in \n"
  "reasonable ways as different from the original version; or \n"
  " \n"
  "d) Limiting the use for publicity purposes of names of licensors or \n"
  "authors of the material; or \n"
  " \n"
  "e) Declining to grant rights under trademark law for use of some \n"
  "trade names, trademarks, or service marks; or \n"
  " \n"
  "f) Requiring indemnification of licensors and authors of that \n"
  "material by anyone who conveys the material (or modified versions of \n"
  "it) with contractual assumptions of liability to the recipient, for \n"
  "any liability that these contractual assumptions directly impose on \n"
  "those licensors and authors. \n"
  " \n"
  "All other non-permissive additional terms are considered 'further \n"
  "restrictions' within the meaning of section 10.  If the Program as you \n"
  "received it, or any part of it, contains a notice stating that it is \n"
  "governed by this License along with a term that is a further \n"
  "restriction, you may remove that term.  If a license document contains \n"
  "a further restriction but permits relicensing or conveying under this \n"
  "License, you may add to a covered work material governed by the terms \n"
  "of that license document, provided that the further restriction does \n"
  "not survive such relicensing or conveying. \n"
  " \n"
  "If you add terms to a covered work in accord with this section, you \n"
  "must place, in the relevant source files, a statement of the \n"
  "additional terms that apply to those files, or a notice indicating \n"
  "where to find the applicable terms. \n"
  " \n"
  "Additional terms, permissive or non-permissive, may be stated in the \n"
  "form of a separately written license, or stated as exceptions; \n"
  "the above requirements apply either way. \n"
  " \n"
  "8. Termination. \n"
  " \n"
  "You may not propagate or modify a covered work except as expressly \n"
  "provided under this License.  Any attempt otherwise to propagate or \n"
  "modify it is void, and will automatically terminate your rights under \n"
  "this License (including any patent licenses granted under the third \n"
  "paragraph of section 11). \n"
  " \n"
  "However, if you cease all violation of this License, then your \n"
  "license from a particular copyright holder is reinstated (a) \n"
  "provisionally, unless and until the copyright holder explicitly and \n"
  "finally terminates your license, and (b) permanently, if the copyright \n"
  "holder fails to notify you of the violation by some reasonable means \n"
  "prior to 60 days after the cessation. \n"
  " \n"
  "Moreover, your license from a particular copyright holder is \n"
  "reinstated permanently if the copyright holder notifies you of the \n"
  "violation by some reasonable means, this is the first time you have \n"
  "received notice of violation of this License (for any work) from that \n"
  "copyright holder, and you cure the violation prior to 30 days after \n"
  "your receipt of the notice. \n"
  " \n"
  "Termination of your rights under this section does not terminate the \n"
  "licenses of parties who have received copies or rights from you under \n"
  "this License.  If your rights have been terminated and not permanently \n"
  "reinstated, you do not qualify to receive new licenses for the same \n"
  "material under section 10. \n"
  " \n"
  "9. Acceptance Not Required for Having Copies. \n"
  " \n"
  "You are not required to accept this License in order to receive or \n"
  "run a copy of the Program.  Ancillary propagation of a covered work \n"
  "occurring solely as a consequence of using peer-to-peer transmission \n"
  "to receive a copy likewise does not require acceptance.  However, \n"
  "nothing other than this License grants you permission to propagate or \n"
  "modify any covered work.  These actions infringe copyright if you do \n"
  "not accept this License.  Therefore, by modifying or propagating a \n"
  "covered work, you indicate your acceptance of this License to do so. \n"
  " \n"
  "10. Automatic Licensing of Downstream Recipients. \n"
  " \n"
  "Each time you convey a covered work, the recipient automatically \n"
  "receives a license from the original licensors, to run, modify and \n"
  "propagate that work, subject to this License.  You are not responsible \n"
  "for enforcing compliance by third parties with this License. \n"
  " \n"
  "An 'entity transaction' is a transaction transferring control of an \n"
  "organization, or substantially all assets of one, or subdividing an \n"
  "organization, or merging organizations.  If propagation of a covered \n"
  "work results from an entity transaction, each party to that \n"
  "transaction who receives a copy of the work also receives whatever \n"
  "licenses to the work the party's predecessor in interest had or could \n"
  "give under the previous paragraph, plus a right to possession of the \n"
  "Corresponding Source of the work from the predecessor in interest, if \n"
  "the predecessor has it or can get it with reasonable efforts. \n"
  " \n"
  "You may not impose any further restrictions on the exercise of the \n"
  "rights granted or affirmed under this License.  For example, you may \n"
  "not impose a license fee, royalty, or other charge for exercise of \n"
  "rights granted under this License, and you may not initiate litigation \n"
  "(including a cross-claim or counterclaim in a lawsuit) alleging that \n"
  "any patent claim is infringed by making, using, selling, offering for \n"
  "sale, or importing the Program or any portion of it. \n"
  " \n"
  "11. Patents. \n"
  " \n"
  "A 'contributor' is a copyright holder who authorizes use under this \n"
  "License of the Program or a work on which the Program is based.  The \n"
  "work thus licensed is called the contributor's 'contributor version'. \n"
  " \n"
  "A contributor's 'essential patent claims' are all patent claims \n"
  "owned or controlled by the contributor, whether already acquired or \n"
  "hereafter acquired, that would be infringed by some manner, permitted \n"
  "by this License, of making, using, or selling its contributor version, \n"
  "but do not include claims that would be infringed only as a \n"
  "consequence of further modification of the contributor version.  For \n"
  "purposes of this definition, 'control' includes the right to grant \n"
  "patent sublicenses in a manner consistent with the requirements of \n"
  "this License. \n"
  " \n"
  "Each contributor grants you a non-exclusive, worldwide, royalty-free \n"
  "patent license under the contributor's essential patent claims, to \n"
  "make, use, sell, offer for sale, import and otherwise run, modify and \n"
  "propagate the contents of its contributor version. \n"
  " \n"
  "In the following three paragraphs, a 'patent license' is any express \n"
  "agreement or commitment, however denominated, not to enforce a patent \n"
  "(such as an express permission to practice a patent or covenant not to \n"
  "sue for patent infringement).  To 'grant' such a patent license to a \n"
  "party means to make such an agreement or commitment not to enforce a \n"
  "patent against the party. \n"
  " \n"
  "If you convey a covered work, knowingly relying on a patent license, \n"
  "and the Corresponding Source of the work is not available for anyone \n"
  "to copy, free of charge and under the terms of this License, through a \n"
  "publicly available network server or other readily accessible means, \n"
  "then you must either (1) cause the Corresponding Source to be so \n"
  "available, or (2) arrange to deprive yourself of the benefit of the \n"
  "patent license for this particular work, or (3) arrange, in a manner \n"
  "consistent with the requirements of this License, to extend the patent \n"
  "license to downstream recipients.  'Knowingly relying' means you have \n"
  "actual knowledge that, but for the patent license, your conveying the \n"
  "covered work in a country, or your recipient's use of the covered work \n"
  "in a country, would infringe one or more identifiable patents in that \n"
  "country that you have reason to believe are valid. \n"
  " \n"
  "If, pursuant to or in connection with a single transaction or \n"
  "arrangement, you convey, or propagate by procuring conveyance of, a \n"
  "covered work, and grant a patent license to some of the parties \n"
  "receiving the covered work authorizing them to use, propagate, modify \n"
  "or convey a specific copy of the covered work, then the patent license \n"
  "you grant is automatically extended to all recipients of the covered \n"
  "work and works based on it. \n"
  " \n"
  "A patent license is 'discriminatory' if it does not include within \n"
  "the scope of its coverage, prohibits the exercise of, or is \n"
  "conditioned on the non-exercise of one or more of the rights that are \n"
  "specifically granted under this License.  You may not convey a covered \n"
  "work if you are a party to an arrangement with a third party that is \n"
  "in the business of distributing software, under which you make payment \n"
  "to the third party based on the extent of your activity of conveying \n"
  "the work, and under which the third party grants, to any of the \n"
  "parties who would receive the covered work from you, a discriminatory \n"
  "patent license (a) in connection with copies of the covered work \n"
  "conveyed by you (or copies made from those copies), or (b) primarily \n"
  "for and in connection with specific products or compilations that \n"
  "contain the covered work, unless you entered into that arrangement, \n"
  "or that patent license was granted, prior to 28 March 2007. \n"
  " \n"
  "Nothing in this License shall be construed as excluding or limiting \n"
  "any implied license or other defenses to infringement that may \n"
  "otherwise be available to you under applicable patent law. \n"
  " \n"
  "12. No Surrender of Others' Freedom. \n"
  " \n"
  "If conditions are imposed on you (whether by court order, agreement or \n"
  "otherwise) that contradict the conditions of this License, they do not \n"
  "excuse you from the conditions of this License.  If you cannot convey a \n"
  "covered work so as to satisfy simultaneously your obligations under this \n"
  "License and any other pertinent obligations, then as a consequence you may \n"
  "not convey it at all.  For example, if you agree to terms that obligate you \n"
  "to collect a royalty for further conveying from those to whom you convey \n"
  "the Program, the only way you could satisfy both those terms and this \n"
  "License would be to refrain entirely from conveying the Program. \n"
  " \n"
  "13. Use with the GNU Affero General Public License. \n"
  " \n"
  "Notwithstanding any other provision of this License, you have \n"
  "permission to link or combine any covered work with a work licensed \n"
  "under version 3 of the GNU Affero General Public License into a single \n"
  "combined work, and to convey the resulting work.  The terms of this \n"
  "License will continue to apply to the part which is the covered work, \n"
  "but the special requirements of the GNU Affero General Public License, \n"
  "section 13, concerning interaction through a network will apply to the \n"
  "combination as such. \n"
  " \n"
  "14. Revised Versions of this License. \n"
  " \n"
  "The Free Software Foundation may publish revised and/or new versions of \n"
  "the GNU General Public License from time to time.  Such new versions will \n"
  "be similar in spirit to the present version, but may differ in detail to \n"
  "address new problems or concerns. \n"
  " \n"
  "Each version is given a distinguishing version number.  If the \n"
  "Program specifies that a certain numbered version of the GNU General \n"
  "Public License 'or any later version' applies to it, you have the \n"
  "option of following the terms and conditions either of that numbered \n"
  "version or of any later version published by the Free Software \n"
  "Foundation.  If the Program does not specify a version number of the \n"
  "GNU General Public License, you may choose any version ever published \n"
  "by the Free Software Foundation. \n"
  " \n"
  "If the Program specifies that a proxy can decide which future \n"
  "versions of the GNU General Public License can be used, that proxy's \n"
  "public statement of acceptance of a version permanently authorizes you \n"
  "to choose that version for the Program. \n"
  " \n"
  "Later license versions may give you additional or different \n"
  "permissions.  However, no additional obligations are imposed on any \n"
  "author or copyright holder as a result of your choosing to follow a \n"
  "later version. \n"
  " \n"
  "15. Disclaimer of Warranty. \n"
  " \n"
  "THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY \n"
  "APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT \n"
  "HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM 'AS IS' WITHOUT WARRANTY \n"
  "OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, \n"
  "THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR \n"
  "PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM \n"
  "IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF \n"
  "ALL NECESSARY SERVICING, REPAIR OR CORRECTION. \n"
  " \n"
  "16. Limitation of Liability. \n"
  " \n"
  "IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING \n"
  "WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS \n"
  "THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY \n"
  "GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE \n"
  "USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF \n"
  "DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD \n"
  "PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), \n"
  "EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF \n"
  "SUCH DAMAGES. \n"
  " \n"
  "17. Interpretation of Sections 15 and 16. \n"
  " \n"
  "If the disclaimer of warranty and limitation of liability provided \n"
  "above cannot be given local legal effect according to their terms, \n"
  "reviewing courts shall apply local law that most closely approximates \n"
  "an absolute waiver of all civil liability in connection with the \n"
  "Program, unless a warranty or assumption of liability accompanies a \n"
  "copy of the Program in return for a fee. \n"
  " \n"
  " END OF TERMS AND CONDITIONS \n\n");
}

__host__ void print_help() {
	printf("Example: ./bin/gpuvmem options [ arguments ...]\n");
	printf("    -h  --help             Shows this\n");
  printf(	"   -X  --blockSizeX       Block X Size for Image (Needs to be pow of 2) (Mandatory)\n");
  printf(	"   -Y  --blockSizeY       Block Y Size for Image (Needs to be pow of 2) (Mandatory)\n");
  printf(	"   -V  --blockSizeV       Block Size for Visibilities (Needs to be pow of 2) (Mandatory)\n");
  printf(	"   -i  --input            The name of the input file of visibilities(MS) (Mandatory)\n");
  printf(	"   -o  --output           The name of the output file of residual visibilities(MS) (Mandatory)\n");
  printf(	"   -O  --output-image     The name of the output image FITS file (Mandatory)\n");
  printf("    -I  --inputdat         The name of the input file of parameters (Mandatory)\n");
  printf("    -m  --modin            mod_in_0 FITS (I_nu_0) image file (Mandatory)\n");
  printf("    -F  --nu_0             Frequency of reference (Mandatory)\n");
  printf("    -a  --alpha_start      Alpha spectral index starting values\n");
  printf("    -x  --minpix           Minimum positive value of a pixel (Optional)\n");
  printf("    -n  --noise            Noise Parameter (Optional)\n");
  printf("    -N  --noise-cut        Noise-cut Parameter (Optional)\n");
  printf("    -l  --lambda           Lambda Regularization Parameter (Optional)\n");
  printf("    -r  --randoms          Percentage of data used when random sampling (Default = 1.0, optional)\n");
  printf("    -P  --prior            Prior used to regularize the solution (Default = 0 = Entropy)\n");
  printf("    -e  --eta              Variable that controls the minimum image value (Default eta = -1.0)\n");
  printf("    -p  --path             MEM path to save FITS images. With last / included. (Example ./../mem/)\n");
  printf("    -f  --file             Output file where final objective function values are saved (Optional)\n");
  printf("    -M  --multigpu         Number of GPUs to use multiGPU image synthesis (Default OFF => 0)\n");
  printf("    -s  --select           If multigpu option is OFF, then select the GPU ID of the GPU you will work on. (Default = 0)\n");
  printf("    -t  --iterations       Number of iterations for optimization (Default = 500)\n");
  printf("    -c  --copyright        Shows copyright conditions\n");
  printf("    -w  --warranty         Shows no warranty details\n");
  printf("        --nopositivity     Run gpuvmem using chi2 with no posititivy restriction\n");
  printf("        --apply-noise      Apply random gaussian noise to visibilities\n");
  printf("        --clipping         Clips the image to positive values\n");
  printf("        --print-images     Prints images per iteration\n");
  printf("        --verbose          Shows information through all the execution\n");
}

__host__ char *strip(const char *string, const char *chars)
{
  char * newstr = (char*)malloc(strlen(string) + 1);
  int counter = 0;

  for ( ; *string; string++) {
    if (!strchr(chars, *string)) {
      newstr[ counter ] = *string;
      ++ counter;
    }
  }

  newstr[counter] = 0;
  return newstr;
}

__host__ Vars getOptions(int argc, char **argv) {
	Vars variables;
  variables.multigpu = "NULL";
  variables.Tin = "NULL";
  variables.ofile = "NULL";
  variables.path = "mem/";
  variables.output_image = "mod_out";
  variables.select = 0;
  variables.blockSizeX = -1;
  variables.blockSizeY = -1;
  variables.blockSizeV = -1;
  variables.nu_0 = -1;
  variables.it_max = 500;
  variables.noise = -1;
  variables.lambda = -1;
  variables.randoms = -1;
  variables.noise_cut = -1;
  variables.minpix = -1;
  variables.reg_term = 0;
  variables.alpha_start = 0.0;
  variables.eta = -1.0;


	long next_op;
	const char* const short_op = "hcwi:o:O:I:m:x:n:N:l:r:f:M:s:p:P:X:Y:V:t:F:a:e:";

	const struct option long_op[] = { //Flag for help, copyright and warranty
                                    {"help", 0, NULL, 'h' },
                                    {"warranty", 0, NULL, 'w' },
                                    {"copyright", 0, NULL, 'c' },
                                    /* These options set a flag. */
                                    {"verbose", 0, &verbose_flag, 1},
                                    {"apply-noise", 0, &apply_noise, 1},
                                    {"nopositivity", 0, &nopositivity, 1},
                                    {"clipping", 0, &clip_flag, 1},
                                    {"print-images", 0, &print_images, 1},
                                    /* These options don’t set a flag. */
                                    {"input", 1, NULL, 'i' }, {"output", 1, NULL, 'o'}, {"output-image", 1, NULL, 'O'},
                                    {"inputdat", 1, NULL, 'I'}, {"modin", 1, NULL, 'm' }, {"noise", 0, NULL, 'n' },
                                    {"lambda", 0, NULL, 'l' }, {"multigpu", 0, NULL, 'M'}, {"select", 1, NULL, 's'},
                                    {"path", 1, NULL, 'p'}, {"prior", 0, NULL, 'P'}, {"eta", 0, NULL, 'e'},
                                    {"blockSizeX", 1, NULL, 'X'}, {"blockSizeY", 1, NULL, 'Y'}, {"blockSizeV", 1, NULL, 'V'},
                                    {"iterations", 0, NULL, 't'}, {"noise-cut", 0, NULL, 'N' }, {"minpix", 0, NULL, 'x' },
                                    {"randoms", 0, NULL, 'r' }, {"nu_0", 1, NULL, 'F' }, {"file", 0, NULL, 'f' },
                                    {"alpha_start", 1, NULL, 'a' }, { NULL, 0, NULL, 0 }};

	if (argc == 1) {
		printf(
				"ERROR. THE PROGRAM HAS BEEN EXECUTED WITHOUT THE NEEDED PARAMETERS OR OPTIONS\n");
		print_help();
		exit(EXIT_SUCCESS);
	}
  int option_index = 0;
	while (1) {
		next_op = getopt_long(argc, argv, short_op, long_op, &option_index);
		if (next_op == -1) {
			break;
		}

		switch (next_op) {
    case 0:
      /* If this option set a flag, do nothing else now. */
      if (long_op[option_index].flag != 0)
        break;
        printf ("option %s", long_op[option_index].name);
      if (optarg)
        printf (" with arg %s", optarg);
        printf ("\n");
        break;
		case 'h':
			print_help();
			exit(EXIT_SUCCESS);
    case 'w':
  		print_warranty();
  		exit(EXIT_SUCCESS);
    case 'c':
    	print_copyright();
    	exit(EXIT_SUCCESS);
		case 'i':
      variables.input = (char*) malloc((strlen(optarg)+1)*sizeof(char));
			strcpy(variables.input, optarg);
			break;
    case 'o':
      variables.output = (char*) malloc((strlen(optarg)+1)*sizeof(char));
  		strcpy(variables.output, optarg);
  		break;
    case 'O':
      variables.output_image = (char*) malloc((strlen(optarg)+1)*sizeof(char));
    	strcpy(variables.output_image, optarg);
    	break;
    case 'I':
      variables.inputdat = (char*) malloc((strlen(optarg)+1)*sizeof(char));
      strcpy(variables.inputdat, optarg);
      break;
    case 'm':
      variables.modin = (char*) malloc((strlen(optarg)+1)*sizeof(char));
    	strcpy(variables.modin, optarg);
    	break;
    case 'x':
      variables.minpix = atof(optarg);
      break;
    case 'e':
      variables.eta = atof(optarg);
      break;
    case 'F':
      variables.nu_0 = atof(optarg);
      break;
    case 'a':
      variables.alpha_start = atof(optarg);
      break;
    case 'n':
      variables.noise = atof(optarg);
      break;
    case 'N':
      variables.noise_cut = atof(optarg);
      break;
    case 'l':
      variables.lambda = atof(optarg);
      break;
    case 'p':
      variables.path = (char*) malloc((strlen(optarg)+1)*sizeof(char));
      strcpy(variables.path, optarg);
      break;
    case 'P':
      variables.reg_term = atoi(optarg);;
      break;
    case 'M':
      variables.multigpu = optarg;
      break;
    case 'r':
      variables.randoms = atof(optarg);
      break;
    case 'f':
      variables.ofile = (char*) malloc((strlen(optarg)+1)*sizeof(char));
      strcpy(variables.ofile, optarg);
      break;
    case 's':
      variables.select = atoi(optarg);
      break;
    case 'X':
      variables.blockSizeX = atoi(optarg);
      break;
    case 'Y':
      variables.blockSizeY = atoi(optarg);
      break;
    case 'V':
      variables.blockSizeV = atoi(optarg);
      break;
    case 't':
      variables.it_max = atoi(optarg);
      break;
		case '?':
			print_help();
			exit(EXIT_FAILURE);
		case -1:
			break;
		default:
      print_help();
			exit(EXIT_FAILURE);
		}
	}

  if(variables.blockSizeX == -1 && variables.blockSizeY == -1 && variables.blockSizeV == -1 ||
     strcmp(strip(variables.input, " "),"") == 0 && strcmp(strip(variables.output, " "),"") == 0 && strcmp(strip(variables.output_image, " "),"") == 0 && strcmp(strip(variables.inputdat, " "),"") == 0 ||
     strcmp(strip(variables.modin, " "),"") == 0 && strcmp(strip(variables.path, " "),"") == 0 || variables.nu_0 == -1 || variables.alpha_start == 0.0) {
        print_help();
        exit(EXIT_FAILURE);
  }

  if(!isPow2(variables.blockSizeX) && !isPow2(variables.blockSizeY) && !isPow2(variables.blockSizeV)){
    print_help();
    exit(EXIT_FAILURE);
  }

  if(variables.randoms > 1.0){
    print_help();
    exit(EXIT_FAILURE);
  }

  if(variables.reg_term > 2){
    print_help();
    exit(EXIT_FAILURE);
  }

  if(strcmp(variables.multigpu,"NULL")!=0 && variables.select != 0){
    print_help();
    exit(EXIT_FAILURE);
  }
	return variables;
}

template <bool nIsPow2>
__global__ void deviceReduceKernel(float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    float mySum = 0.f;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}





__host__ float deviceReduce(float *in, long N)
{
  float *device_out;
  gpuErrchk(cudaMalloc(&device_out, sizeof(float)*1024));
  gpuErrchk(cudaMemset(device_out, 0, sizeof(float)*1024));

  int threads = 512;
  int blocks = min((int(NearestPowerOf2(N)) + threads - 1) / threads, 1024);
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  bool isPower2 = isPow2(N);
  if(isPower2){
    deviceReduceKernel<true><<<blocks, threads, smemSize>>>(in, device_out, N);
    gpuErrchk(cudaDeviceSynchronize());
  }else{
    deviceReduceKernel<false><<<blocks, threads, smemSize>>>(in, device_out, N);
    gpuErrchk(cudaDeviceSynchronize());
  }

  float *h_odata = (float *) malloc(blocks*sizeof(float));
  float sum = 0.0;

  gpuErrchk(cudaMemcpy(h_odata, device_out, blocks * sizeof(float),cudaMemcpyDeviceToHost));
  for (int i=0; i<blocks; i++)
  {
    sum += h_odata[i];
  }
  cudaFree(device_out);
  free(h_odata);
	return sum;
}

__global__ void hermitianSymmetry(float *Ux, float *Vx, cufftComplex *Vo, float freq, int numVisibilities)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities){
      if(Ux[i] < 0.0){
        Ux[i] *= -1.0;
        Vx[i] *= -1.0;
        Vo[i].y *= -1.0;
      }
      Ux[i] = (Ux[i] * freq) / LIGHTSPEED;
      Vx[i] = (Vx[i] * freq) / LIGHTSPEED;
  }
}


__device__ float attenuation(float beam_fwhm, float beam_freq, float beam_cutoff, float freq, float xobs, float yobs, float DELTAX, float DELTAY)
{

		int j = threadIdx.x + blockDim.x * blockIdx.x;
		int i = threadIdx.y + blockDim.y * blockIdx.y;

    float atten_result;

    int x0 = xobs;
    int y0 = yobs;
    float x = (j - x0) * DELTAX * RPDEG;
    float y = (i - y0) * DELTAY * RPDEG;

    float arc = sqrtf(x*x+y*y);
    float c = 4.0*logf(2.0);
    float a = (beam_fwhm*beam_freq/(freq*1e-9));
    float r = arc/a;
    float atten = expf(-c*r*r);
    if(arc <= beam_cutoff){
      atten_result = atten;
    }else{
      atten_result = 0.0;
    }

    return atten_result;
}



__global__ void total_attenuation(float *total_atten, float beam_fwhm, float beam_freq, float beam_cutoff, float freq, float xobs, float yobs, float DELTAX, float DELTAY, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  float attenPerFreq = attenuation(beam_fwhm, beam_freq, beam_cutoff, freq, xobs, yobs, DELTAX, DELTAY);

  total_atten[N*i+j] += attenPerFreq;
}

__global__ void mean_attenuation(float *total_atten, int channels, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  total_atten[N*i+j] /= channels;
}

__global__ void weight_image(float *weight_image, float *total_atten, float noise_jypix, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  float atten = total_atten[N*i+j];
  weight_image[N*i+j] += (atten / noise_jypix) * (atten / noise_jypix);
}

__global__ void noise_image(float *noise_image, float *weight_image, float noise_jypix, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  float noiseval;
  noiseval = sqrtf(1.0/weight_image[N*i+j]);
  noise_image[N*i+j] = noiseval;
}

__global__ void apply_beam(float beam_fwhm, float beam_freq, float beam_cutoff, cufftComplex *image, long N, float xobs, float yobs, float fg_scale, float freq, float DELTAX, float DELTAY)
{
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.y + blockDim.y * blockIdx.y;


    float dx = DELTAX * 60.0;
    float dy = DELTAY * 60.0;
    float x = (j - xobs) * dx;
    float y = (i - yobs) * dy;
    float arc = RPARCM*sqrtf(x*x+y*y);
    float c = 4.0*logf(2.0);
    float a = (beam_fwhm*beam_freq/(freq*1e-9));
    float r = arc/a;
    float atten = expf(-c*r*r);

    if(arc <= beam_cutoff){
      image[N*i+j].x = image[N*i+j].x * fg_scale * atten;
      //image[N*i+j].x = image[N*i+j].x * atten;
      image[N*i+j].y = 0.0;
    }else{
      image[N*i+j].x = 0.0;
      image[N*i+j].y = 0.0;
    }


}


/*--------------------------------------------------------------------
 * Phase rotate the visibility data in "image" to refer phase to point
 * (x,y) instead of (0,0).
 * Multiply pixel V(i,j) by exp(-2 pi i (x/ni + y/nj))
 *--------------------------------------------------------------------*/
__global__ void phase_rotate(cufftComplex *data, long M, long N, float xphs, float yphs)
{

		int j = threadIdx.x + blockDim.x * blockIdx.x;
		int i = threadIdx.y + blockDim.y * blockIdx.y;

    float u,v, phase, c, s, re, im;
    float du = xphs/(float)M;
    float dv = yphs/(float)N;

    if(j < M/2){
      u = du * j;
    }else{
      u = du * (j-M);
    }

    if(i < N/2){
      v = dv * i;
    }else{
      v = dv * (i-N);
    }

    phase = 2.0*(u+v);
    #if (__CUDA_ARCH__ >= 300 )
      sincospif(phase, &s, &c);
    #else
      c = cospif(phase);
      s = sinpif(phase);
    #endif
    re = data[N*i+j].x;
    im = data[N*i+j].y;
    data[N*i+j].x = re * c - im * s;
    data[N*i+j].y = re * s + im * c;
}


/*
 * Interpolate in the visibility array to find the visibility at (u,v);
 */
__global__ void vis_mod(cufftComplex *Vm, cufftComplex *Vo, cufftComplex *V, float *Ux, float *Vx, float deltau, float deltav, long numVisibilities, long N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  long i1, i2, j1, j2;
  float du, dv, u, v;
  float v11, v12, v21, v22;
  float Zreal;
  float Zimag;

  if (i < numVisibilities){

    u = Ux[i]/deltau;
    v = Vx[i]/deltav;

    if (fabsf(u) > (N/2)+0.5 || fabsf(v) > (N/2)+0.5) {
      printf("Error in residual: u,v = %f,%f\n", u, v);
      asm("trap;");
    }

    if(u < 0.0){
      u = N + u;
    }

    if(v < 0.0){
      v = N + v;
    }

    i1 = u;
    i2 = (i1+1)%N;
    du = u - i1;
    j1 = v;
    j2 = (j1+1)%N;
    dv = v - j1;

    if (i1 < 0 || i1 > N || j1 < 0 || j2 > N) {
      printf("Error in residual: u,v = %f,%f, %ld,%ld, %ld,%ld\n", u, v, i1, i2, j1, j2);
      asm("trap;");
    }

    /* Bilinear interpolation: real part */
    v11 = V[N*j1 + i1].x; /* [i1, j1] */
    v12 = V[N*j2 + i1].x; /* [i1, j2] */
    v21 = V[N*j1 + i2].x; /* [i2, j1] */
    v22 = V[N*j2 + i2].x; /* [i2, j2] */
    Zreal = (1-du)*(1-dv)*v11 + (1-du)*dv*v12 + du*(1-dv)*v21 + du*dv*v22;
    /* Bilinear interpolation: imaginary part */
    v11 = V[N*j1 + i1].y; /* [i1, j1] */
    v12 = V[N*j2 + i1].y; /* [i1, j2] */
    v21 = V[N*j1 + i2].y; /* [i2, j1] */
    v22 = V[N*j2 + i2].y; /* [i2, j2] */
    Zimag = (1-du)*(1-dv)*v11 + (1-du)*dv*v12 + du*(1-dv)*v21 + du*dv*v22;

    Vm[i].x = Zreal;
    Vm[i].y = Zimag;

  }

}


__global__ void residual(cufftComplex *Vr, cufftComplex *Vm, cufftComplex *Vo, long numVisibilities){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < numVisibilities){
    Vr[i].x = Vm[i].x - Vo[i].x;
    Vr[i].y = Vm[i].y - Vo[i].y;
  }
}
__global__ void clipWNoise(float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX, float eta)

{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  if(noise[N*i+j] > noise_cut){
    if(eta > 0.0){
      I[N*i+j].x = 0.0;
    }
    else{
      I[N*i+j].x = -1.0 * eta * MINPIX;
    }

  }

  I[N*i+j].y = 0;
}

__global__ void clip2IWNoise(float *noise, float2 *I, long N, float noise_cut, float minpix, float fg_scale, float eta)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  if(noise[N*i+j] > noise_cut){
    if(eta > 0.0){
      I[N*i+j].x = 0.0;
    }
    else{
      I[N*i+j].x = -1.0 * eta * minpix * fg_scale;;
    }

  }

}

__global__ void clip2I(float2 *I, long N, float minpix, float fg_scale)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  //I_nu_0
  if(I[N*i+j].x < minpix * fg_scale){
      I[N*i+j].x = minpix * fg_scale;
  }
  /*//alpha
  if(I[N*i+j].y < minpix_alpha){
      I[N*i+j].y = minpix_alpha;
  }*/
}

__global__ void clip(cufftComplex *I, long N, float MINPIX)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  //T
  if(I[N*i+j].x < MINPIX && MINPIX >= 0.0){
      I[N*i+j].x = MINPIX;
  }
  I[N*i+j].y = 0;
}

__global__ void newP(float2 *p, float2 *xi, float xmin, long N, float minpix, float fg_scale, float eta)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[N*i+j].x *= xmin;
  xi[N*i+j].y *= xmin;
  //I_nu_0
  if(p[N*i+j].x + xi[N*i+j].x > -1.0*eta*minpix*fg_scale){
    p[N*i+j].x += xi[N*i+j].x;
  }else{
    p[N*i+j].x = -1.0*eta*minpix*fg_scale;
    xi[N*i+j].x = 0.0;
  }

  p[N*i+j].y += xi[N*i+j].y;
  /*//alpha
  if(p[N*i+j].y + xi[N*i+j].y > minpix_alpha){
    p[N*i+j].y += xi[N*i+j].y;
  }else{
    p[N*i+j].y = minpix_alpha;
    xi[N*i+j].y = 0.0;
  }*/

}

__global__ void newPNoPositivity(float2 *p, float2 *xi, float xmin, long N)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[N*i+j].x *= xmin;
  xi[N*i+j].y *= xmin;

  p[N*i+j].x += xi[N*i+j].x;
  p[N*i+j].y += xi[N*i+j].y;
}

__global__ void evaluateXt(float2 *xt, float2 *pcom, float2 *xicom, float x, long N, float minpix, float fg_scale, float eta)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
  //I_nu_0
  if(pcom[N*i+j].x + x * xicom[N*i+j].x > -1.0*eta*minpix*fg_scale){
    xt[N*i+j].x = pcom[N*i+j].x + x * xicom[N*i+j].x;
  }else{
    xt[N*i+j].x = -1.0*eta*minpix*fg_scale;
  }

  xt[N*i+j].y = pcom[N*i+j].y + x * xicom[N*i+j].y;
  //alpha
  /*if(pcom[N*i+j].y + x * xicom[N*i+j].y > minpix_alpha){
    xt[N*i+j].y = pcom[N*i+j].y + x * xicom[N*i+j].y;
  }else{
      xt[N*i+j].y = minpix_alpha;
  }*/
}


__global__ void evaluateXtNoPositivity(float2 *xt, float2 *pcom, float2 *xicom, float x, long N)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
  //I_nu_0
  xt[N*i+j].x = pcom[N*i+j].x + x * xicom[N*i+j].x;
  //alpha
  xt[N*i+j].y = pcom[N*i+j].y + x * xicom[N*i+j].y;
}


__global__ void chi2Vector(float *chi2, cufftComplex *Vr, float *w, long numVisibilities)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < numVisibilities){
		chi2[i] =  w[i] * ((Vr[i].x * Vr[i].x) + (Vr[i].y * Vr[i].y));
	}

}

__global__ void SVector(float *S, float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX, float eta)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float entropy = 0.0;
  if(noise[N*i+j] <= noise_cut){
    entropy = I[N*i+j].x * logf((I[N*i+j].x / MINPIX) + (eta + 1.0));
  }

  S[N*i+j] = entropy;
}

__global__ void QVector(float *Q, float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float entropy = 0.0;
  if(noise[N*i+j] <= noise_cut){
    if((i>0 && i<N) && (j>0 && j<N)){
      entropy = (I[N*i+j].x - I[N*i+(j-1)].x) * (I[N*i+j].x - I[N*i+(j-1)].x) + (I[N*i+j].x - I[N*i+(j+1)].x) * (I[N*i+j].x - I[N*i+(j+1)].x) + (I[N*i+j].x - I[N*(i-1)+j].x) * (I[N*i+j].x - I[N*(i-1)+j].x) + (I[N*i+j].x - I[N*(i+1)+j].x) * (I[N*i+j].x - I[N*(i+1)+j].x);
      entropy /= 2;
    }else{
      entropy = I[N*i+j].x;
    }
  }

  Q[N*i+j] = entropy;
}

__global__ void TVVector(float *TV, float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float tv = 0.0;
  if(noise[N*i+j] <= noise_cut){
    if(i!= N-1 || j!=N-1){
      float dx = I[N*i+(j+1)].x - I[N*i+j].x;
      float dy = I[N*(i+1)+j].x - I[N*i+j].x;
      tv = sqrtf((dx * dx) + (dy * dy));
    }else{
      tv = 0;
    }
  }

  TV[N*i+j] = tv;
}
__global__ void searchDirection(float2 *g, float2 *xi, float2 *h, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  g[N*i+j].x = -xi[N*i+j].x;
  g[N*i+j].y = -xi[N*i+j].y;

  xi[N*i+j].x = h[N*i+j].x = g[N*i+j].x;
  xi[N*i+j].y = h[N*i+j].y = g[N*i+j].y;
}

__global__ void newXi(float2 *g, float2 *xi, float2 *h, float gam, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  g[N*i+j].x = -xi[N*i+j].x;
  g[N*i+j].y = -xi[N*i+j].y;

  xi[N*i+j].x = h[N*i+j].x = g[N*i+j].x + gam * h[N*i+j].x;
  xi[N*i+j].y = h[N*i+j].y = g[N*i+j].y + gam * h[N*i+j].y;
}

__global__ void getGGandDGG(float *gg, float *dgg, float2 *xi, float2 *g, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float gg_Inu_0, gg_alpha;
  float dgg_Inu_0, dgg_alpha;

  gg_Inu_0 = g[N*i+j].x * g[N*i+j].x;
  gg_alpha = g[N*i+j].y * g[N*i+j].y;

  dgg_Inu_0 = (xi[N*i+j].x + g[N*i+j].x) * xi[N*i+j].x;
  dgg_alpha = (xi[N*i+j].y + g[N*i+j].y) * xi[N*i+j].y;

  gg[N*i+j] = gg_Inu_0 + gg_alpha;
  dgg[N*i+j] = dgg_Inu_0 + dgg_alpha;
}

__global__ void restartDPhi(float2 *dChi2, float *dS, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  //I_nu_0
  dChi2[N*i+j].x = 0.0;
  //alpha
  dChi2[N*i+j].y = 0.0;

  dS[N*i+j] = 0.0;

}

__global__ void changeAlpha(float2 *I, float new_alpha, long N)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;


	I[N*i+j].y = new_alpha; 
}

__global__ void DS(float *dH, cufftComplex *I, float *noise, float noise_cut, float lambda, float MINPIX, float eta, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  if(noise[N*i+j] <= noise_cut){
    dH[N*i+j] = lambda * (logf((I[N*i+j].x / MINPIX) + (eta+1.0)) + 1.0/(1.0 + (((eta+1.0)*MINPIX) / I[N*i+j].x)));
  }
}

__global__ void DQ(float *dQ, cufftComplex *I, float *noise, float noise_cut, float lambda, float MINPIX, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  if(noise[N*i+j] <= noise_cut){
    if((i>0 && i<N) && (j>0 && j<N)){
    dQ[N*i+j] = (I[N*i+j].x - I[N*i+(j-1)].x) + (I[N*i+j].x - I[N*i+(j+1)].x) + (I[N*i+j].x - I[N*(i-1)+j].x)  + (I[N*i+j].x - I[N*(i+1)+j].x);
  }else{
    dQ[N*i+j] = I[N*i+j].x;
    }
    dQ[N*i+j] *= lambda;
  }
}

__global__ void DTV(float *dTV, cufftComplex *I, float *noise, float noise_cut, float lambda, float MINPIX, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dtv = 0.0;
  float num = 0.0;
  float den = 0.0;
  if(noise[N*i+j] <= noise_cut){
    if(i!= N-1 || j!=N-1){
      float a = I[N*i+(j+1)].x;
      float b = I[N*(i+1)+j].x;
      float y = I[N*i+j].x;
      float num = -a-b+(2*y);
      float den = (a*a) - 2*y*(a+b) + (b*b) + 2*(y*y);
      if(den <= 0){
        dtv = 0;
      }else{
        dtv = num/sqrtf(den);
      }
    }else{
      dtv = 0;
    }
    dTV[N*i+j] = lambda * dtv;
  }
}

__global__ void DChi2(float *noise, float *dChi2, cufftComplex *Vr, float *U, float *V, float *w, long N, long numVisibilities, float fg_scale, float noise_cut, float xobs, float yobs, float DELTAX, float DELTAY, float beam_fwhm, float beam_freq, float beam_cutoff, float freq)
{

	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  int x0 = xobs;
  int y0 = yobs;
  float x = (j - x0) * DELTAX * RPDEG;
  float y = (i - y0) * DELTAY * RPDEG;

	float Ukv, Vkv, cosk, sink, atten;

  atten = attenuation(beam_fwhm, beam_freq, beam_cutoff, freq, xobs, yobs, DELTAX, DELTAY);

  float dchi2 = 0.0;
  if(noise[N*i+j] <= noise_cut){
  	for(int v=0; v<numVisibilities; v++){
      Ukv = x * U[v];
  		Vkv = y * V[v];
      #if (__CUDA_ARCH__ >= 300 )
        sincospif(2.0*(Ukv+Vkv), &sink, &cosk);
      #else
        cosk = cospif(2.0*(Ukv+Vkv));
        sink = sinpif(2.0*(Ukv+Vkv));
      #endif
      dchi2 += w[v]*((Vr[v].x * cosk) - (Vr[v].y * sink));
  	}

  dchi2 *= fg_scale * atten;
  dChi2[N*i+j] = dchi2;
  }
}



__global__ void DChi2_total_alpha(float *noise, float2 *dchi2_total, float *dchi2, float2 *I, float *dS, float lambda, float nu, float nu_0, float noise_cut, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu_0, alpha, dalpha, dI_nu_0;
  float nudiv = nu/nu_0;

  I_nu_0 = I[N*i+j].x;
  alpha = I[N*i+j].y;

  dI_nu_0 = powf(nudiv, alpha);
  dalpha = dI_nu_0 * I_nu_0 * logf(nudiv);
  /*if (i==242 & j==277)
    printf("nu : %e, dalpha : %e\n", nu, dalpha);*/

  if(noise[N*i+j] <= noise_cut){
    if(lambda != 0.0)
    {
      dchi2_total[N*i+j].x += (dchi2[N*i+j] + dS[N*i+j]) * dI_nu_0 * 0.f;
      dchi2_total[N*i+j].y += (dchi2[N*i+j] + dS[N*i+j]) * dalpha;
    }else{
      dchi2_total[N*i+j].x += dchi2[N*i+j] * dI_nu_0 * 0.f;
      dchi2_total[N*i+j].y += dchi2[N*i+j] * dalpha;
    }
  }else{
    dchi2_total[N*i+j].x += 0.f;
    dchi2_total[N*i+j].y += 0.f;
  }
}

__global__ void DChi2_total_I_nu_0(float *noise, float2 *dchi2_total, float *dchi2, float2 *I, float *dS, float lambda, float nu, float nu_0, float noise_cut, long N)
{

	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu_0, alpha, dalpha, dI_nu_0;
  float nudiv = nu/nu_0;

  I_nu_0 = I[N*i+j].x;
  alpha = I[N*i+j].y;

  dI_nu_0 = powf(nudiv, alpha);
  dalpha = dI_nu_0 * I_nu_0 * logf(nudiv);

  /*if (i==242 & j==277)
    printf("nu : %e, dalpha : %e\n", nu, dalpha);*/


  if(noise[N*i+j] <= noise_cut){
    if(lambda != 0.0)
    {
      dchi2_total[N*i+j].x += (dchi2[N*i+j] + dS[N*i+j]) * dI_nu_0;
      dchi2_total[N*i+j].y += (dchi2[N*i+j] + dS[N*i+j]) * dalpha * 0.f;
    }else{
      dchi2_total[N*i+j].x += dchi2[N*i+j] * dI_nu_0;
      dchi2_total[N*i+j].y += dchi2[N*i+j] * dalpha * 0.f;
    }
  }else{
    dchi2_total[N*i+j].x += 0.f;
    dchi2_total[N*i+j].y += 0.f;
  }
}


__global__ void calculateInu(cufftComplex *I_nu, float2 *image2, float nu, float nu_0, float fg_scale, float minpix, float eta, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu_0, alpha, nudiv_pow_alpha, nudiv;

  nudiv = nu/nu_0;

  I_nu_0 = image2[N*i+j].x;
  alpha = image2[N*i+j].y;

  nudiv_pow_alpha = powf(nudiv, alpha);

  I_nu[N*i+j].x = I_nu_0 * nudiv_pow_alpha / fg_scale;

  if(I_nu[N*i+j].x < -1.0*eta*minpix){
    I_nu[N*i+j].x = -1.0*eta*minpix;
  }

  I_nu[N*i+j].y = 0.f;
}


__host__ float chiCuadrado(float2 *I)
{
  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
    cudaSetDevice(firstgpu);
  }

  float resultPhi = 0.0;
  float resultchi2  = 0.0;
  float resultS  = 0.0;

  if(clip_flag){
    clip2I<<<numBlocksNN, threadsPerBlockNN>>>(I, N, MINPIX, fg_scale);
    gpuErrchk(cudaDeviceSynchronize());
  }

  clip2IWNoise<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, I, N, noise_cut, MINPIX, fg_scale, eta);
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemset(device_S, 0, sizeof(float)*M*N));


  if(num_gpus == 1){
    cudaSetDevice(selected);
    for(int f=0; f<data.nfields;f++){
      for(int i=0; i<data.total_frequencies;i++){

        if(fields[f].numVisibilitiesPerFreq[i] != 0){

          calculateInu<<<numBlocksNN, threadsPerBlockNN>>>(device_Inu, I, fields[f].visibilities[i].freq, nu_0, fg_scale, MINPIX, eta, N);
          gpuErrchk(cudaDeviceSynchronize());

          if(clip_flag){
            clip<<<numBlocksNN, threadsPerBlockNN>>>(device_Inu, N, MINPIX);
            gpuErrchk(cudaDeviceSynchronize());
          }

          clipWNoise<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_Inu, N, noise_cut, MINPIX, eta);
          gpuErrchk(cudaDeviceSynchronize());


          if(iter>0 && lambda!=0.0){
            switch(reg_term){
              case 0:
                SVector<<<numBlocksNN, threadsPerBlockNN>>>(device_S, device_noise_image, device_Inu, N, noise_cut, MINPIX, eta);
                gpuErrchk(cudaDeviceSynchronize());
                break;
              case 1:
                QVector<<<numBlocksNN, threadsPerBlockNN>>>(device_S, device_noise_image, device_Inu, N, noise_cut, MINPIX);
                gpuErrchk(cudaDeviceSynchronize());
                break;
              case 2:
                TVVector<<<numBlocksNN, threadsPerBlockNN>>>(device_S, device_noise_image, device_Inu, N, noise_cut, MINPIX);
                gpuErrchk(cudaDeviceSynchronize());
                break;
              default:
                printf("Selected prior is not defined\n");
                goToError();
                break;
            }
          }


        	apply_beam<<<numBlocksNN, threadsPerBlockNN>>>(beam_fwhm, beam_freq, beam_cutoff, device_Inu, N, fields[f].global_xobs, fields[f].global_yobs, fg_scale, fields[f].visibilities[i].freq, DELTAX, DELTAY);
        	gpuErrchk(cudaDeviceSynchronize());

        	//FFT 2D
        	if ((cufftExecC2C(plan1GPU, (cufftComplex*)device_Inu, (cufftComplex*)device_V, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
        		printf("CUFFT exec error\n");
        		goToError();
        	}
        	gpuErrchk(cudaDeviceSynchronize());

          //PHASE_ROTATE
          phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(device_V, M, N, fields[f].global_xobs, fields[f].global_yobs);
        	gpuErrchk(cudaDeviceSynchronize());

          //RESIDUAL CALCULATION
          vis_mod<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vm, fields[f].device_visibilities[i].Vo, device_V, fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, deltau, deltav, fields[f].numVisibilitiesPerFreq[i], N);
        	gpuErrchk(cudaDeviceSynchronize());

          residual<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].Vm, fields[f].device_visibilities[i].Vo, fields[f].numVisibilitiesPerFreq[i]);
          gpuErrchk(cudaDeviceSynchronize());


        	////chi 2 VECTOR
        	chi2Vector<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(vars_per_field[f].device_vars[i].chi2, fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].weight, fields[f].numVisibilitiesPerFreq[i]);
        	gpuErrchk(cudaDeviceSynchronize());

        	//REDUCTIONS
        	//chi2
        	resultchi2  += deviceReduce(vars_per_field[f].device_vars[i].chi2, fields[f].numVisibilitiesPerFreq[i]);
          //S
          resultS += deviceReduce(device_S, M*N);
        }
      }
    }
  }else{
    for(int f=0; f<data.nfields; f++){
      #pragma omp parallel for schedule(static,1)
      for (int i = 0; i < data.total_frequencies; i++)
  		{
        float partial_chi2 = 0.0;
        float partial_S = 0.0;
        unsigned int j = omp_get_thread_num();
  			//unsigned int num_cpu_threads = omp_get_num_threads();
  			// set and check the CUDA device for this CPU thread
  			int gpu_id = -1;
  			cudaSetDevice((i%num_gpus) + firstgpu);   // "% num_gpus" allows more CPU threads than GPU devices
  			cudaGetDevice(&gpu_id);
        if(fields[f].numVisibilitiesPerFreq[i] != 0){


          calculateInu<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].device_vars[i].device_Inu, I, fields[f].visibilities[i].freq, nu_0, fg_scale, MINPIX, eta, N);
          gpuErrchk(cudaDeviceSynchronize());

          if(clip_flag){
            clip<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].device_vars[i].device_Inu, N, MINPIX);
            gpuErrchk(cudaDeviceSynchronize());
          }

          clipWNoise<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, vars_per_field[f].device_vars[i].device_Inu, N, noise_cut, MINPIX, eta);
          gpuErrchk(cudaDeviceSynchronize());

          if(iter>0 && lambda!=0.0){
            switch(reg_term){
              case 0:
                SVector<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].device_vars[i].device_S, device_noise_image, vars_per_field[f].device_vars[i].device_Inu, N, noise_cut, MINPIX, eta);
                gpuErrchk(cudaDeviceSynchronize());
                break;
              case 1:
                QVector<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].device_vars[i].device_S, device_noise_image, vars_per_field[f].device_vars[i].device_Inu, N, noise_cut, MINPIX);
                gpuErrchk(cudaDeviceSynchronize());
                break;
              case 2:
                TVVector<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].device_vars[i].device_S, device_noise_image, vars_per_field[f].device_vars[i].device_Inu, N, noise_cut, MINPIX);
                gpuErrchk(cudaDeviceSynchronize());
                break;
              default:
                printf("Selected prior is not defined\n");
                goToError();
                break;
            }
          }

        	apply_beam<<<numBlocksNN, threadsPerBlockNN>>>(beam_fwhm, beam_freq, beam_cutoff, vars_per_field[f].device_vars[i].device_Inu, N, fields[f].global_xobs, fields[f].global_yobs, fg_scale, fields[f].visibilities[i].freq, DELTAX, DELTAY);
        	gpuErrchk(cudaDeviceSynchronize());


        	//FFT 2D
        	if ((cufftExecC2C(vars_per_field[f].device_vars[i].plan, (cufftComplex*)vars_per_field[f].device_vars[i].device_Inu, (cufftComplex*)vars_per_field[f].device_vars[i].device_V, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
        		printf("CUFFT exec error\n");
        		//return -1 ;
        		goToError();
        	}
        	gpuErrchk(cudaDeviceSynchronize());

          //PHASE_ROTATE
          phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].device_vars[i].device_V, M, N, fields[f].global_xobs, fields[f].global_yobs);
        	gpuErrchk(cudaDeviceSynchronize());

          //RESIDUAL CALCULATION
          vis_mod<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vm, fields[f].device_visibilities[i].Vo, vars_per_field[f].device_vars[i].device_V, fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, deltau, deltav, fields[f].numVisibilitiesPerFreq[i], N);
        	gpuErrchk(cudaDeviceSynchronize());


          residual<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].Vm, fields[f].device_visibilities[i].Vo, fields[f].numVisibilitiesPerFreq[i]);
          gpuErrchk(cudaDeviceSynchronize());

        	////chi2 VECTOR
        	chi2Vector<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(vars_per_field[f].device_vars[i].chi2, fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].weight, fields[f].numVisibilitiesPerFreq[i]);
        	gpuErrchk(cudaDeviceSynchronize());


          partial_chi2 = deviceReduce(vars_per_field[f].device_vars[i].chi2, fields[f].numVisibilitiesPerFreq[i]);

          partial_S = deviceReduce(vars_per_field[f].device_vars[i].device_S, M*N);;
        	//REDUCTIONS
        	//chi2
          #pragma omp critical
          {
            resultchi2  += partial_chi2;
            resultS  += partial_S;
          }
        }
      }
    }
  }
  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
    cudaSetDevice(firstgpu);
  }

  resultPhi = (0.5 * resultchi2) + (lambda * resultS);

  final_chi2 = resultchi2;
  final_H = resultS;
  /*printf("chi2 value = %.5f\n", resultchi2);
  printf("H value = %.5f\n", resultH);
  printf("(1/2) * chi2 value = %.5f\n", 0.5*resultchi2);
  printf("lambda * H value = %.5f\n", lambda*resultH);
  printf("Phi value = %.5f\n\n", resultPhi);*/

  return resultPhi;
}



__host__ void dchiCuadrado(float2 *I, float2 *dxi2)
{

  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
    cudaSetDevice(firstgpu);
  }

  if(clip_flag){
    clip2I<<<numBlocksNN, threadsPerBlockNN>>>(I, N, MINPIX, fg_scale);
    gpuErrchk(cudaDeviceSynchronize());
  }

  gpuErrchk(cudaMemset(device_dchi2_total, 0, sizeof(float2)*M*N));

  restartDPhi<<<numBlocksNN, threadsPerBlockNN>>>(device_dchi2_total, device_dS, N);
  gpuErrchk(cudaDeviceSynchronize());

  if(print_images)
    float2toImage(I, mod_in, out_image, mempath, iter, M, N, 1);

  if(num_gpus == 1){
    cudaSetDevice(selected);
    for(int f=0; f<data.nfields; f++){
      for(int i=0; i<data.total_frequencies;i++){
        if(fields[f].numVisibilitiesPerFreq[i] != 0){

            calculateInu<<<numBlocksNN, threadsPerBlockNN>>>(device_Inu, I, fields[f].visibilities[i].freq, nu_0, fg_scale, MINPIX, eta, N);
            gpuErrchk(cudaDeviceSynchronize());

            if(clip_flag){
              clip<<<numBlocksNN, threadsPerBlockNN>>>(device_Inu, N, MINPIX);
              gpuErrchk(cudaDeviceSynchronize());
            }

            if(iter>0 && lambda!=0.0){
              switch(reg_term){
                case 0:
                  DS<<<numBlocksNN, threadsPerBlockNN>>>(device_dS, device_Inu, device_noise_image, noise_cut, lambda, MINPIX, eta, N);
                  gpuErrchk(cudaDeviceSynchronize());
                  break;
                case 1:
                  DQ<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].device_vars[i].device_S, vars_per_field[f].device_vars[i].device_Inu, device_noise_image, noise_cut, lambda, MINPIX, N);
                  gpuErrchk(cudaDeviceSynchronize());
                  break;
                case 2:
                  DTV<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].device_vars[i].device_S, vars_per_field[f].device_vars[i].device_Inu, device_noise_image, noise_cut, lambda, MINPIX, N);
                  gpuErrchk(cudaDeviceSynchronize());
                  break;
                default:
                  printf("Selected prior is not defined\n");
                  goToError();
                  break;
              }
            }

            DChi2<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, vars_per_field[f].device_vars[i].dchi2, fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].weight, N, fields[f].numVisibilitiesPerFreq[i], fg_scale, noise_cut, fields[f].global_xobs, fields[f].global_yobs, DELTAX, DELTAY, beam_fwhm, beam_freq, beam_cutoff, fields[f].visibilities[i].freq);
            gpuErrchk(cudaDeviceSynchronize());

            if(flag_opt==1){
              DChi2_total_I_nu_0<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_dchi2_total, vars_per_field[f].device_vars[i].dchi2, I, device_dS, lambda, fields[f].visibilities[i].freq, nu_0, noise_cut, N);
            	gpuErrchk(cudaDeviceSynchronize());
            }else{
              DChi2_total_alpha<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_dchi2_total, vars_per_field[f].device_vars[i].dchi2, I, device_dS, lambda, fields[f].visibilities[i].freq, nu_0, noise_cut, N);
            	gpuErrchk(cudaDeviceSynchronize());
            }

        }
      }
    }
  }else{
    for(int f=0;f<data.nfields;f++){
      #pragma omp parallel for schedule(static,1)
      for (int i = 0; i < data.total_frequencies; i++)
      {
        unsigned int j = omp_get_thread_num();
        //unsigned int num_cpu_threads = omp_get_num_threads();
        // set and check the CUDA device for this CPU thread
        int gpu_id = -1;
        cudaSetDevice((i%num_gpus) + firstgpu);   // "% num_gpus" allows more CPU threads than GPU devices
        cudaGetDevice(&gpu_id);

        gpuErrchk(cudaMemset(vars_per_field[f].device_vars[i].device_S, 0, sizeof(float)*M*N));


        calculateInu<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].device_vars[i].device_Inu, I, fields[f].visibilities[i].freq, nu_0, fg_scale, MINPIX, eta, N);
        gpuErrchk(cudaDeviceSynchronize());


        if(clip_flag){
          clip<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].device_vars[i].device_Inu, N, MINPIX);
          gpuErrchk(cudaDeviceSynchronize());
        }

        if(iter>0 && lambda!=0.0){
          switch(reg_term){
            case 0:
              DS<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].device_vars[i].device_S, vars_per_field[f].device_vars[i].device_Inu, device_noise_image, noise_cut, lambda, MINPIX, eta, N);
              gpuErrchk(cudaDeviceSynchronize());
              break;
            case 1:
              DQ<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].device_vars[i].device_S, vars_per_field[f].device_vars[i].device_Inu, device_noise_image, noise_cut, lambda, MINPIX, N);
              gpuErrchk(cudaDeviceSynchronize());
              break;
            case 2:
              DTV<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].device_vars[i].device_S, vars_per_field[f].device_vars[i].device_Inu, device_noise_image, noise_cut, lambda, MINPIX, N);
              gpuErrchk(cudaDeviceSynchronize());
              break;
            default:
              printf("Selected prior is not defined\n");
              goToError();
              break;
          }
        }

        if(fields[f].numVisibilitiesPerFreq[i] != 0){
          DChi2<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, vars_per_field[f].device_vars[i].dchi2, fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].weight, N, fields[f].numVisibilitiesPerFreq[i], fg_scale, noise_cut, fields[f].global_xobs, fields[f].global_yobs, DELTAX, DELTAY, beam_fwhm, beam_freq, beam_cutoff, fields[f].visibilities[i].freq);
          gpuErrchk(cudaDeviceSynchronize());

          #pragma omp critical
          {
            if(flag_opt==1){
              DChi2_total_I_nu_0<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_dchi2_total, vars_per_field[f].device_vars[i].dchi2, I, vars_per_field[f].device_vars[i].device_S, lambda, fields[f].visibilities[i].freq, nu_0, noise_cut, N);
            	gpuErrchk(cudaDeviceSynchronize());
            }else{
              DChi2_total_alpha<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_dchi2_total, vars_per_field[f].device_vars[i].dchi2, I, vars_per_field[f].device_vars[i].device_S, lambda, fields[f].visibilities[i].freq, nu_0, noise_cut, N);
            	gpuErrchk(cudaDeviceSynchronize());
            }

          }
        }
      }
    }
  }

  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
    cudaSetDevice(firstgpu);
  }

  //DPhi<<<numBlocksNN, threadsPerBlockNN>>>(device_dphi, device_dchi2_total, N);
  //gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy2D(dxi2, sizeof(float2), device_dchi2_total, sizeof(float2), sizeof(float2), M*N, cudaMemcpyDeviceToDevice));

}
