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

  Additionally, this program uses some NVIDIA routines whose copyright is held
  by NVIDIA end user license agreement (EULA).

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

namespace cg = cooperative_groups;

extern long M, N;
extern int numVisibilities, iterations, iterthreadsVectorNN, blocksVectorNN, nopositivity, crpix1, crpix2, \
status_mod_in, verbose_flag, clip_flag, num_gpus, selected, iter, t_telescope, multigpu, firstgpu, reg_term, apply_noise, print_images, gridding;

extern cufftHandle plan1GPU;
extern cufftComplex *device_I, *device_V, *device_fg_image, *device_image;

extern float *device_dphi, *device_chi2, *device_dchi2, *device_S, *device_dchi2_total, *device_dS, *device_noise_image;
extern float noise_jypix, fg_scale, DELTAX, DELTAY, deltau, deltav, noise_cut, MINPIX, \
minpix, lambda, ftol, random_probability, final_chi2, final_S, eta;

extern dim3 threadsPerBlockNN, numBlocksNN;

extern float beam_noise, beam_bmaj, beam_bmin, b_noise_aux, antenna_diameter, pb_factor, pb_cutoff;
extern double ra, dec;

extern MSData data;

extern char* mempath, *out_image;

extern fitsfile *mod_in;

extern Field *fields;

extern VariablesPerField *vars_per_field;

extern varsPerGPU *vars_gpu;


// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

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
                antenna_diameter = 1.4; /* CBI2 Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 90.0*RPARCM; /* radians */
                break;
        case 2:
                antenna_diameter = 12.0; /* ALMA Antenna Diameter */
                pb_factor = 1.13; /* FWHM Factor */
                pb_cutoff = 1.0*RPARCM; /* radians */
                break;
        case 3:
                antenna_diameter = 22.0; /* ATCA Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 1.0*RPARCM; /* radians */
                break;
        case 4:
                antenna_diameter = 25.0; /* VLA Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 20.0*RPARCM; /* radians */
                break;
        case 5:
                antenna_diameter = 3.5; /* SZA Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 20.0*RPARCM; /* radians */
                break;
        case 6:
                antenna_diameter = 0.9; /* CBI Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 20.0*RPARCM; /* radians */
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
  printf(	"   -X  --blockSizeX       Block X Size for Image (Needs to be pow of 2)\n");
  printf(	"   -Y  --blockSizeY       Block Y Size for Image (Needs to be pow of 2)\n");
  printf(	"   -V  --blockSizeV       Block Size for Visibilities (Needs to be pow of 2)\n");
  printf(	"   -i  --input            The name of the input file of visibilities(MS)\n");
  printf(	"   -o  --output           The name of the output file of residual visibilities(MS)\n");
  printf(	"   -O  --output-image     The name of the output image FITS file\n");
  printf("    -I  --inputdat         The name of the input file of parameters\n");
  printf("    -m  --modin            mod_in_0 FITS file\n");
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
  printf("    -g  --gridding         Use gridding to decrease the number of visibilities. This is done in CPU (Need to select the CPU threads that will grid the input visibilities)\n");
  printf("    -c  --copyright        Shows copyright conditions\n");
  printf("    -w  --warranty         Shows no warranty details\n");
  printf("        --xcorr            Run gpuvmem with cross-correlation\n");
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
  variables.ofile = "NULL";
  variables.path = "mem/";
  variables.output_image = "mod_out.fits";
  variables.select = 0;
  variables.blockSizeX = -1;
  variables.blockSizeY = -1;
  variables.blockSizeV = -1;
  variables.it_max = 500;
  variables.noise = -1;
  variables.lambda = -1;
  variables.randoms = 1.0;
  variables.noise_cut = -1;
  variables.minpix = -1;
  variables.reg_term = 0;
  variables.eta = -1.0;
  variables.gridding = 0;


	long next_op;
	const char* const short_op = "hcwi:o:O:I:m:x:n:N:l:r:f:M:s:e:p:P:X:Y:V:t:g:";

	const struct option long_op[] = { //Flag for help, copyright and warranty
                                    {"help", 0, NULL, 'h' },
                                    {"warranty", 0, NULL, 'w' },
                                    {"copyright", 0, NULL, 'c' },
                                    /* These options set a flag. */
                                    {"verbose", 0, &verbose_flag, 1},
                                    {"nopositivity", 0, &nopositivity, 1},
                                    {"clipping", 0, &clip_flag, 1},
                                    {"apply-noise", 0, &apply_noise, 1},
                                    {"print-images", 0, &print_images, 1},
                                    /* These options don’t set a flag. */
                                    {"input", 1, NULL, 'i' }, {"output", 1, NULL, 'o'}, {"output-image", 1, NULL, 'O'},
                                    {"inputdat", 1, NULL, 'I'}, {"modin", 1, NULL, 'm' }, {"noise", 0, NULL, 'n' },
                                    {"lambda", 0, NULL, 'l' }, {"multigpu", 0, NULL, 'M'}, {"select", 1, NULL, 's'},
                                    {"path", 1, NULL, 'p'}, {"prior", 0, NULL, 'P'}, {"eta", 0, NULL, 'e'},
                                    {"blockSizeX", 1, NULL, 'X'}, {"blockSizeY", 1, NULL, 'Y'}, {"blockSizeV", 1, NULL, 'V'},
                                    {"iterations", 0, NULL, 't'}, {"noise-cut", 0, NULL, 'N' }, {"minpix", 0, NULL, 'x' },
                                    {"randoms", 0, NULL, 'r' }, {"file", 0, NULL, 'f' }, {"gridding", 0, NULL, 'g' }, { NULL, 0, NULL, 0 }};

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
    case 'n':
      variables.noise = atof(optarg);
      break;
    case 'e':
      variables.eta = atof(optarg);
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
    case 'g':
      variables.gridding = atoi(optarg);
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
     strcmp(strip(variables.modin, " "),"") == 0 && strcmp(strip(variables.path, " "),"") == 0) {
        print_help();
        exit(EXIT_FAILURE);
  }

  if(!isPow2(variables.blockSizeX) && !isPow2(variables.blockSizeY) && !isPow2(variables.blockSizeV)){
    print_help();
    exit(EXIT_FAILURE);
  }

  if(variables.reg_term > 3){
    print_help();
    exit(EXIT_FAILURE);
  }

  if(variables.randoms > 1.0){
    print_help();
    exit(EXIT_FAILURE);
  }

  if(variables.gridding < 0){
    print_help();
    exit(EXIT_FAILURE);
  }

  if(strcmp(variables.multigpu,"NULL")!=0 && variables.select != 0){
    print_help();
    exit(EXIT_FAILURE);
  }
	return variables;
}



#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

__host__ void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    gpuErrchk(cudaGetDevice(&device));
    gpuErrchk(cudaGetDeviceProperties(&prop, device));


    threads = (n < maxThreads*2) ? NearestPowerOf2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    blocks = MIN(maxBlocks, blocks);
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void deviceReduceKernel(T *g_idata, T *g_odata, unsigned int n)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  T mySum = 0;

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
  cg::sync(cta);


  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256))
  {
      sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  cg::sync(cta);

  if ((blockSize >= 256) &&(tid < 128))
  {
          sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid <  64))
  {
     sdata[tid] = mySum = mySum + sdata[tid +  64];
  }

  cg::sync(cta);

#if (__CUDA_ARCH__ >= 300 )
  if ( tid < 32 )
  {
      cg::coalesced_group active = cg::coalesced_threads();

      // Fetch final intermediate sum from 2nd warp
      if (blockSize >=  64) mySum += sdata[tid + 32];
      // Reduce final warp using shuffle
      for (int offset = warpSize/2; offset > 0; offset /= 2)
      {
           mySum += active.shfl_down(mySum, offset);
      }
  }
#else
  // fully unroll reduction within a single warp
  if ((blockSize >=  64) && (tid < 32))
  {
      sdata[tid] = mySum = mySum + sdata[tid + 32];
  }

  cg::sync(cta);

  if ((blockSize >=  32) && (tid < 16))
  {
      sdata[tid] = mySum = mySum + sdata[tid + 16];
  }

  cg::sync(cta);

  if ((blockSize >=  16) && (tid <  8))
  {
      sdata[tid] = mySum = mySum + sdata[tid +  8];
  }

  cg::sync(cta);

  if ((blockSize >=   8) && (tid <  4))
  {
      sdata[tid] = mySum = mySum + sdata[tid +  4];
  }

  cg::sync(cta);

  if ((blockSize >=   4) && (tid <  2))
  {
      sdata[tid] = mySum = mySum + sdata[tid +  2];
  }

  cg::sync(cta);

  if ((blockSize >=   2) && ( tid <  1))
  {
      sdata[tid] = mySum = mySum + sdata[tid +  1];
  }

  cg::sync(cta);
#endif

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = mySum;
}





template <class T>
__host__ T deviceReduce(T *in, long N)
{
  T *device_out;

  int maxThreads = 256;
  int maxBlocks = NearestPowerOf2(N)/maxThreads;

  int threads = 0;
  int blocks = 0;

  getNumBlocksAndThreads(N, maxBlocks, maxThreads, blocks, threads);

  //printf("N %d, threads: %d, blocks %d\n", N, threads, blocks);
  //threads = maxThreads;
  //blocks = NearestPowerOf2(N)/threads;

  gpuErrchk(cudaMalloc(&device_out, sizeof(T)*blocks));
  gpuErrchk(cudaMemset(device_out, 0, sizeof(T)*blocks));

  int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  bool isPower2 = isPow2(N);

  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  if(isPower2){
    switch (threads){
      case 512:
        deviceReduceKernel<T, 512, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 256:
        deviceReduceKernel<T, 256, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 128:
        deviceReduceKernel<T, 128, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 64:
        deviceReduceKernel<T, 64, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 32:
        deviceReduceKernel<T, 32, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 16:
        deviceReduceKernel<T, 16, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 8:
        deviceReduceKernel<T, 8, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 4:
        deviceReduceKernel<T, 4, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 2:
        deviceReduceKernel<T, 2, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 1:
        deviceReduceKernel<T, 1, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
    }
  }else{
    switch (threads){
      case 512:
        deviceReduceKernel<T, 512, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 256:
        deviceReduceKernel<T, 256, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 128:
        deviceReduceKernel<T, 128, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 64:
        deviceReduceKernel<T, 64, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 32:
        deviceReduceKernel<T, 32, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 16:
        deviceReduceKernel<T, 16, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 8:
        deviceReduceKernel<T, 8, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 4:
        deviceReduceKernel<T, 4, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 2:
        deviceReduceKernel<T, 2, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 1:
        deviceReduceKernel<T, 1, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
    }
  }

  T *h_odata = (T *) malloc(blocks*sizeof(T));
  T sum = 0;

  gpuErrchk(cudaMemcpy(h_odata, device_out, blocks * sizeof(T),cudaMemcpyDeviceToHost));
  for (int i=0; i<blocks; i++)
  {
    sum += h_odata[i];
  }
  cudaFree(device_out);
  free(h_odata);
	return sum;
}

template <typename T, typename C>
__host__
void fftShift_2D(T* data, C* w, C* u, C* v, int M, int N)
{
    // 2D Slice & 1D Line
    int sLine = N;
    int sSlice = M * N;

    // Transformations Equations
    int sEq1 = (sSlice + sLine) / 2;
    int sEq2 = (sSlice - sLine) / 2;

    for(int i = 0; i < M; i++){
      for(int j = 0; j < N; j++){
        // Thread Index (2D)
        int xIndex =  j;
        int yIndex = i;

        // Thread Index Converted into 1D Index
        int index = (yIndex * N) + xIndex;

        T regTemp;
        C uTemp;
        C vTemp;
        C wTemp;

        if (xIndex < N / 2)
        {
          if (yIndex < M / 2)
          {
            regTemp = data[index];
            uTemp = u[index];
            vTemp = v[index];
            wTemp = w[index];

            // First Quad
            data[index] = data[index + sEq1];
            u[index] = u[index + sEq1];
            v[index] = v[index + sEq1];
            w[index] = w[index + sEq1];

            // Third Quad
            data[index + sEq1] = regTemp;
            u[index + sEq1] = uTemp;
            v[index + sEq1] = vTemp;
            w[index + sEq1] = wTemp;
          }
        }
        else
        {
          if (yIndex < M / 2)
          {
            regTemp = data[index];
            uTemp = u[index];
            vTemp = v[index];
            wTemp = w[index];

            // Second Quad
            data[index] = data[index + sEq2];
            v[index] = u[index + sEq2];
            u[index] = v[index + sEq2];
            w[index] = w[index + sEq2];

            // Fourth Quad
            data[index + sEq2] = regTemp;
            u[index + sEq2] = uTemp;
            v[index + sEq2] = vTemp;
            w[index + sEq2] = wTemp;
          }
        }
      }
    }
}

__host__ void do_gridding(Field *fields, MSData *data, float deltau, float deltav, int M, int N, int *total_visibilities)
{
  for(int f=0; f<data->nfields; f++){
  	for(int i=0; i < data->total_frequencies; i++){
      for(int sto=0; sto < data->nstokes; sto++){
        #pragma omp parallel for schedule(static,1)
        for(int z=0; z < fields[f].numVisibilitiesPerFreqPerStoke[i][sto]; z++){
          int k,j;
          float u, v, w;
          cufftComplex Vo;

          u  = fields[f].visibilities[i][sto].u[z];
          v =  fields[f].visibilities[i][sto].v[z];
          Vo = fields[f].visibilities[i][sto].Vo[z];
          w = fields[f].visibilities[i][sto].weight[z];
          //Correct scale and apply hermitian symmetry (it will be applied afterwards)
          if(u < 0.0){
            u *= -1.0;
            v *= -1.0;
            Vo.y *= -1.0;
          }

          u *= fields[f].nu[i] / LIGHTSPEED;
          v *= fields[f].nu[i] / LIGHTSPEED;

          j = roundf(u/fabsf(deltau) + N/2);
      		k = roundf(v/fabsf(deltav) + M/2);

          #pragma omp critical
          {
            if(k < M && j < N){
                fields[f].gridded_visibilities[i][sto].Vo[N*k+j].x += w*Vo.x;
                fields[f].gridded_visibilities[i][sto].Vo[N*k+j].y += w*Vo.y;
                fields[f].gridded_visibilities[i][sto].weight[N*k+j] += w;
            }
          }
        }

        int visCounter = 0;
        #pragma omp parallel for schedule(static,1)
        for(int k=0; k<M; k++){
          for(int j=0; j<N; j++){
            float deltau_meters = fabsf(deltau) * (LIGHTSPEED/fields[f].nu[i]);
            float deltav_meters = fabsf(deltav) * (LIGHTSPEED/fields[f].nu[i]);

            float u_meters = (j - (N/2)) * deltau_meters;
            float v_meters = (k - (M/2)) * deltav_meters;

            fields[f].gridded_visibilities[i][sto].u[N*k+j] = u_meters;
            fields[f].gridded_visibilities[i][sto].v[N*k+j] = v_meters;

            float weight = fields[f].gridded_visibilities[i][sto].weight[N*k+j];
            if(weight > 0.0f){
              fields[f].gridded_visibilities[i][sto].Vo[N*k+j].x /= weight;
              fields[f].gridded_visibilities[i][sto].Vo[N*k+j].y /= weight;
              #pragma omp critical
              {
                  visCounter++;
              }
            }else{
              fields[f].gridded_visibilities[i][sto].weight[N*k+j] = 0.0f;
            }
          }
        }

        fields[f].visibilities[i][sto].u = (float*)realloc(fields[f].visibilities[i][sto].u, visCounter*sizeof(float));
        fields[f].visibilities[i][sto].v = (float*)realloc(fields[f].visibilities[i][sto].v, visCounter*sizeof(float));

        fields[f].visibilities[i][sto].Vo = (cufftComplex*)realloc(fields[f].visibilities[i][sto].Vo, visCounter*sizeof(cufftComplex));

        fields[f].visibilities[i][sto].Vm = (cufftComplex*)malloc(visCounter*sizeof(cufftComplex));
        memset(fields[f].visibilities[i][sto].Vm, 0, visCounter*sizeof(cufftComplex));

        fields[f].visibilities[i][sto].weight = (float*)realloc(fields[f].visibilities[i][sto].weight, visCounter*sizeof(float));

        int l = 0;
        for(int k=0; k<M; k++){
          for(int j=0; j<N; j++){
            float weight = fields[f].gridded_visibilities[i][sto].weight[N*k+j];
            if(weight > 0.0f){
              fields[f].visibilities[i][sto].u[l] = fields[f].gridded_visibilities[i][sto].u[N*k+j];
              fields[f].visibilities[i][sto].v[l] = fields[f].gridded_visibilities[i][sto].v[N*k+j];
              fields[f].visibilities[i][sto].Vo[l].x = fields[f].gridded_visibilities[i][sto].Vo[N*k+j].x;
              fields[f].visibilities[i][sto].Vo[l].y = fields[f].gridded_visibilities[i][sto].Vo[N*k+j].y;
              fields[f].visibilities[i][sto].weight[l] = fields[f].gridded_visibilities[i][sto].weight[N*k+j];
              l++;
            }
          }
        }

        free(fields[f].gridded_visibilities[i][sto].u);
        free(fields[f].gridded_visibilities[i][sto].v);
        free(fields[f].gridded_visibilities[i][sto].Vo);
        free(fields[f].gridded_visibilities[i][sto].weight);

        if(fields[f].numVisibilitiesPerFreqPerStoke[i][sto] > 0){
          fields[f].numVisibilitiesPerFreqPerStoke[i][sto] = visCounter;
          *total_visibilities += visCounter;
        }
      }
    }
  }

  int local_max = 0;
  int max = 0;
  for(int f=0; f < data->nfields; f++){
    for(int i=0; i< data->total_frequencies; i++){
      local_max = *std::max_element(fields[f].numVisibilitiesPerFreqPerStoke[i],fields[f].numVisibilitiesPerFreqPerStoke[i]+data->nstokes);
      if(local_max > max){
        max = local_max;
      }
    }
  }
}

__host__ float calculateNoise(Field *fields, MSData data, int *total_visibilities, int blockSizeV)
{
  //Declaring block size and number of blocks for visibilities
  float sum_inverse_weight = 0.0;
  float sum_weights = 0.0;
  long UVpow2;

  for(int f=0; f<data.nfields; f++){
  	for(int i=0; i< data.total_frequencies; i++){
      for(int sto=0; sto < data.nstokes; sto++){
        //Calculating beam noise
        for(int j=0; j< fields[f].numVisibilitiesPerFreqPerStoke[i][sto]; j++){
          if(fields[f].visibilities[i][sto].weight[j] >= 0.0){
            sum_inverse_weight += 1/fields[f].visibilities[i][sto].weight[j];
            sum_weights += fields[f].visibilities[i][sto].weight[j];
          }
        }
        if(verbose_flag){
          printf("Field: %d, Freq :%d - %f, Stoke: %d - Vis: %d\n", f, i, fields[f].nu[i], sto, fields[f].numVisibilitiesPerFreqPerStoke[i][sto]);
        }
        *total_visibilities += fields[f].numVisibilitiesPerFreqPerStoke[i][sto];
        UVpow2 = NearestPowerOf2(fields[f].numVisibilitiesPerFreqPerStoke[i][sto]);
        fields[f].visibilities[i][sto].threadsPerBlockUV = blockSizeV;
        fields[f].visibilities[i][sto].numBlocksUV = UVpow2/fields[f].visibilities[i][sto].threadsPerBlockUV;
      }
    }
  }


  if(verbose_flag){
      float aux_noise = sqrt(sum_inverse_weight)/ *total_visibilities;
      printf("Calculated NOISE %e\n", aux_noise);
      printf("Using canvas NOISE anyway...\n");
      printf("Canvas NOISE = %e\n", beam_noise);
  }

  if(beam_noise == -1){
      beam_noise = sqrt(sum_inverse_weight)/ *total_visibilities;
      if(verbose_flag){
        printf("No NOISE value detected in canvas...\n");
        printf("Using NOISE: %e ...\n", beam_noise);
      }
  }

  return sum_weights;
}
/*__global__ void do_gridding(float *u, float *v, cufftComplex *Vo, cufftComplex *Vo_g, float *w, float *w_g, int* count, float deltau, float deltav, int visibilities, int M, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < visibilities)
	{
		int k, j;
		j = roundf(u[i]/deltau + M/2);
		k = roundf(v[i]/deltav + N/2);

		if (k < M && j < N)
		{

			atomicAdd(&Vo_g[N*k+j].x, Vo[i].x);
			atomicAdd(&Vo_g[N*k+j].y, Vo[i].y);
      atomicAdd(&w_g[N*k+j], (1.0/w[i]));
      atomicAdd(&count[N*k*j], 1);
		}
	}
}

__global__ void calculateCoordinates(float *u_g, float *v_g, float deltau, float deltav, int M, int N)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;

  u_g[N*i+j] = j*deltau - (N/2)*deltau;
  v_g[N*i+j] = i*deltav - (N/2)*deltav;
}

__global__ void calculateAvgVar(cufftComplex *V_g, float *w_g, int *count, int M, int N)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;

  int counter = count[N*i+j];
  if(counter > 0){
    V_g[N*i+j].x = V_g[N*i+j].x / counter;
    V_g[N*i+j].y = V_g[N*i+j].y / counter;
    w_g[N*i+j] = counter / w_g[N*i+j];
  }else{
    w_g[N*i+j] = 0.0;
  }
}*/

__global__ void hermitianSymmetry(float *Ux, float *Vx, cufftComplex *Vo, float freq, int numVisibilities)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities){
      if(Ux[i] < 0.0){
        Ux[i] *= -1.0;
        Vx[i] *= -1.0;
        Vo[i].y *= -1.0;
      }
      Ux[i] *= freq / LIGHTSPEED;
      Vx[i] *= freq / LIGHTSPEED;
  }
}

__device__ float AiryDiskBeam(float distance, float lambda, float antenna_diameter, float pb_factor)
{
        float atten;
        float r = pb_factor * lambda / antenna_diameter;
        float bessel_arg = PI*distance/(r/RZ);
        float bessel_func = j1f(bessel_arg);
        if(distance == 0.0f) {
                atten = 1.0f;
        }else{
                atten = 4.0f * (bessel_func/bessel_arg) * (bessel_func/bessel_arg);
        }
        return atten;
}

__device__ float GaussianBeam(float distance, float lambda, float antenna_diameter, float pb_factor)
{
        float fwhm = pb_factor * lambda / antenna_diameter;
        float c = 4.0*logf(2.0);
        float r = distance/fwhm;
        float atten = expf(-c*r*r);
        return atten;
}

__device__ float attenuation(float antenna_diameter, float pb_factor, float pb_cutoff, float freq, float xobs, float yobs, float DELTAX, float DELTAY)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float atten_result, atten;

        int x0 = xobs;
        int y0 = yobs;
        float x = (j - x0) * DELTAX * RPDEG;
        float y = (i - y0) * DELTAY * RPDEG;

        float arc = sqrtf(x*x+y*y);
        float lambda = LIGHTSPEED/freq;

        atten = GaussianBeam(arc, lambda, antenna_diameter, pb_factor);

        if(arc <= pb_cutoff) {
                atten_result = atten;
        }else{
                atten_result = 0.0f;
        }

        return atten_result;
}



__global__ void total_attenuation(float *total_atten, float antenna_diameter, float pb_factor, float pb_cutoff, float freq, float xobs, float yobs, float DELTAX, float DELTAY, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float attenPerFreq = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, xobs, yobs, DELTAX, DELTAY);

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

__global__ void apply_beam(float antenna_diameter, float pb_factor, float pb_cutoff, cufftComplex *fg_image, cufftComplex *image, long N, float xobs, float yobs, float fg_scale, float freq, float DELTAX, float DELTAY)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, xobs, yobs, DELTAX, DELTAY);

        image[N*i+j].x = fg_image[N*i+j].x * atten * fg_scale;
        //image[N*i+j].x = image[N*i+j].x * atten;
        image[N*i+j].y = 0.0;
}


/*--------------------------------------------------------------------
 * Phase rotate the visibility data in "image" to refer phase to point
 * (x,y) instead of (0,0).
 * Multiply pixel V(i,j) by exp(2 pi i (x/ni + y/nj))
 *--------------------------------------------------------------------*/
__global__ void phase_rotate(cufftComplex *data, long M, long N, float xphs, float yphs)
{

		int j = threadIdx.x + blockDim.x * blockIdx.x;
		int i = threadIdx.y + blockDim.y * blockIdx.y;

    float u,v, phase, c, s, re, im;
    float du = xphs/(float)M;
    float dv = yphs/(float)N;

    if(j < M/2 + 1){
      u = du * j;
    }else{
      u = du * (j-M);
    }

    if(i < N/2 + 1){
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
 __global__ void vis_mod(cufftComplex *Vm, cufftComplex *V, float *Ux, float *Vx, float *weight, float deltau, float deltav, long numVisibilities, long N)
 {
         int i = threadIdx.x + blockDim.x * blockIdx.x;
         long i1, i2, j1, j2;
         float du, dv, u, v;
         cufftComplex v11, v12, v21, v22;
         float Zreal;
         float Zimag;

         if (i < numVisibilities) {

                 u = Ux[i]/deltau;
                 v = Vx[i]/deltav;

                 if (fabsf(u) <= (N/2)+0.5 && fabsf(v) <= (N/2)+0.5) {

                         if(u < 0.0) {
                                 u += N;
                         }

                         if(v < 0.0) {
                                 v += N;
                         }

                         i1 = u;
                         i2 = (i1+1)%N;
                         du = u - i1;
                         j1 = v;
                         j2 = (j1+1)%N;
                         dv = v - j1;

                         if (i1 >= 0 && i1 < N && i2 >= 0 && i2 < N && j1 >= 0 && j1 < N && j2 >= 0 && j2 < N) {
                                 /* Bilinear interpolation */
                                 v11 = V[N*j1 + i1]; /* [i1, j1] */
                                 v12 = V[N*j2 + i1]; /* [i1, j2] */
                                 v21 = V[N*j1 + i2]; /* [i2, j1] */
                                 v22 = V[N*j2 + i2]; /* [i2, j2] */

                                 Zreal = (1-du)*(1-dv)*v11.x + (1-du)*dv*v12.x + du*(1-dv)*v21.x + du*dv*v22.x;
                                 Zimag = (1-du)*(1-dv)*v11.y + (1-du)*dv*v12.y + du*(1-dv)*v21.y + du*dv*v22.y;

                                 Vm[i].x = Zreal;
                                 Vm[i].y = Zimag;
                         }else{
                                 weight[i] = 0.0f;
                         }
                 }else{
                         //Vm[i].x = 0.0f;
                         //Vm[i].y = 0.0f;
                         weight[i] = 0.0f;
                 }

         }

 }


__global__ void residual(cufftComplex *Vr, cufftComplex *Vm, cufftComplex *Vo, long numVisibilities){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < numVisibilities){
    Vr[i].x = Vm[i].x - Vo[i].x;
    Vr[i].y = Vm[i].y - Vo[i].y;
  }
}



__global__ void clipWNoise(cufftComplex *fg_image, float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX, float eta)
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

  fg_image[N*i+j].x = I[N*i+j].x;
  fg_image[N*i+j].y = 0;
}


__global__ void getGandDGG(float *gg, float *dgg, float *xi, float *g, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  gg[N*i+j] = g[N*i+j] * g[N*i+j];
  dgg[N*i+j] = (xi[N*i+j] + g[N*i+j]) * xi[N*i+j];
}

__global__ void clip(cufftComplex *I, long N, float MINPIX)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  if(I[N*i+j].x < MINPIX && MINPIX >= 0.0){
      I[N*i+j].x = MINPIX;
  }
  I[N*i+j].y = 0;
}

__global__ void newP(cufftComplex *p, float *xi, float xmin, float MINPIX, float eta, long N)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[N*i+j] *= xmin;
  if(p[N*i+j].x + xi[N*i+j] > -1.0*eta*MINPIX){
    p[N*i+j].x += xi[N*i+j];
  }else{
    p[N*i+j].x = -1.0*eta*MINPIX;
    xi[N*i+j] = 0.0;
  }
  p[N*i+j].y = 0.0;
}

__global__ void newPNoPositivity(cufftComplex *p, float *xi, float xmin, long N)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[N*i+j] *= xmin;
  p[N*i+j].x += xi[N*i+j];
  p[N*i+j].y = 0.0;
}

__global__ void evaluateXt(cufftComplex *xt, cufftComplex *pcom, float *xicom, float x, float MINPIX, float eta, long N)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  if(pcom[N*i+j].x + x * xicom[N*i+j] > -1.0*eta*MINPIX){
    xt[N*i+j].x = pcom[N*i+j].x + x * xicom[N*i+j];
  }else{
      xt[N*i+j].x = -1.0*eta*MINPIX;
  }
  xt[N*i+j].y = 0.0;
}

__global__ void evaluateXtNoPositivity(cufftComplex *xt, cufftComplex *pcom, float *xicom, float x, long N)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  xt[N*i+j].x = pcom[N*i+j].x + x * xicom[N*i+j];
  xt[N*i+j].y = 0.0;
}


__global__ void chi2Vector(float *chi2, cufftComplex *Vr, float *w, long numVisibilities)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < numVisibilities){
    if(w[i] > 0.0){
		    chi2[i] =  w[i] * ((Vr[i].x * Vr[i].x) + (Vr[i].y * Vr[i].y));
    }else{
        chi2[i] = 0.0f;
    }
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

__global__ void QPVector(float *Q, float *noise, cufftComplex *I, long N, float noise_cut)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float qp = 0.0;
  if(noise[N*i+j] <= noise_cut){
    if((i>0 && i<N) && (j>0 && j<N)){
      qp = (I[N*i+j].x - I[N*i+(j-1)].x) * (I[N*i+j].x - I[N*i+(j-1)].x) +
           (I[N*i+j].x - I[N*i+(j+1)].x) * (I[N*i+j].x - I[N*i+(j+1)].x) +
           (I[N*i+j].x - I[N*(i-1)+j].x) * (I[N*i+j].x - I[N*(i-1)+j].x) +
           (I[N*i+j].x - I[N*(i+1)+j].x) * (I[N*i+j].x - I[N*(i+1)+j].x);
      qp /= 2.0;
    }else{
      qp = I[N*i+j].x;
    }
  }

  Q[N*i+j] = qp;
}

__global__ void TVVector(float *TV, float *noise, cufftComplex *I, long N, float noise_cut)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float tv = 0.0;
  if(noise[N*i+j] <= noise_cut){
    if(i!= N-1 || j!=N-1){
      float dx = I[N*i+j].x - I[N*i+(j+1)].x;
      float dy = I[N*i+j].x - I[N*(i+1)+j].x;
      tv = sqrtf((dx * dx) + (dy * dy));
    }else{
      tv = I[N*i+j].x;
    }
  }

  TV[N*i+j] = tv;
}

__global__ void LVector(float *L, float *noise, cufftComplex *I, long N, float noise_cut)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  float Dx, Dy;

  if(noise[N*i+j] <= noise_cut){
    if((i>1 && i<N-1) && (j>1 && j<N-1)){
      Dx = I[N*i+(j-1)].x - 2 * I[N*i+j].x + I[N*i+(j+1)].x;
      Dy = I[N*(i-1)+j].x - 2 * I[N*i+j].x + I[N*(i+1)+j].x;
      L[N*i+j] = 0.5 * (Dx + Dy) * (Dx + Dy);
    }else{
      L[N*i+j] = I[N*i+j].x;
    }
  }
}

__global__ void searchDirection(float *g, float *xi, float *h, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  g[N*i+j] = -xi[N*i+j];
  xi[N*i+j] = h[N*i+j] = g[N*i+j];
}

__global__ void searchDirection_LBFGS(float *xi, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[N*i+j] = -xi[N*i+j];
}

__global__ void newXi(float *g, float *xi, float *h, float gam, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  g[N*i+j] = -xi[N*i+j];
  xi[N*i+j] = h[N*i+j] = g[N*i+j] + gam * h[N*i+j];
}

__global__ void DS(float *dH, cufftComplex *I, float *noise, float noise_cut, float lambda, float MINPIX, float eta, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  if(noise[N*i+j] <= noise_cut){
    if(I[N*i+j].x != 0.0){
      dH[N*i+j] = lambda * (logf((I[N*i+j].x / MINPIX) + (eta+1.0)) + 1.0/(1.0 + (((eta+1.0)*MINPIX) / I[N*i+j].x)));
    }else{
      dH[N*i+j] = lambda * logf((I[N*i+j].x / MINPIX));
    }
  }
}

__global__ void DQ(float *dQ, cufftComplex *I, float *noise, float noise_cut, float lambda, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  if(noise[N*i+j] <= noise_cut){
    if((i>0 && i<N-1) && (j>0 && j<N-1)){
      dQ[N*i+j] = 2 * (4 * I[N*i+j].x - (I[N*(i+1)+j].x + I[N*(i-1)+j].x + I[N*i+(j+1)].x + I[N*i+(j-1)].x));
    }else{
      dQ[N*i+j] = I[N*i+j].x;
    }
    dQ[N*i+j] *= lambda;
  }
}

__global__ void DTV(float *dTV, cufftComplex *I, float *noise, float noise_cut, float lambda, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float num0, num1, num2;
  float den0, den1, den2;
  float dtv;

  if(noise[N*i+j] <= noise_cut){
    if((i>0 && i<N-1) && (j>0 && j<N-1)){
      num0 = 2 * I[N*i+j].x - I[N*i+(j+1)].x - I[N*(i+1)+j].x;
      num1 = I[N*i+j].x - I[N*i+(j-1)].x;
      num2 = I[N*i+j].x - I[N*(i-1)+j].x;

      den0 = (I[N*i+j].x - I[N*i+(j+1)].x) * (I[N*i+j].x - I[N*i+(j+1)].x) +
             (I[N*i+j].x - I[N*(i+1)+j].x) * (I[N*i+j].x - I[N*(i+1)+j].x);

      den1 = (I[N*i+(j-1)].x - I[N*i+j].x) * (I[N*i+(j-1)].x - I[N*i+j].x) +
             (I[N*i+(j-1)].x - I[N*(i+1)+(j-1)].x) * (I[N*i+(j-1)].x - I[N*(i+1)+(j-1)].x);

      den2 = (I[N*(i-1)+j].x - I[N*(i-1)+(j+1)].x) * (I[N*(i-1)+j].x - I[N*(i-1)+(j+1)].x) +
             (I[N*(i-1)+j].x - I[N*i+j].x) * (I[N*(i-1)+j].x - I[N*i+j].x);
      if(den0 == 0 || den1 == 0 || den2 == 0){
        dtv = I[N*i+j].x;
      }else{
        dtv = num0/sqrtf(den0) + num1/sqrtf(den1) + num2/sqrtf(den2);
      }
    }else{
      dtv = I[N*i+j].x;
    }
    dTV[N*i+j] = lambda * dtv;
  }
}

__global__ void DL(float *dL, cufftComplex *I, float *noise, float noise_cut, float lambda, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  if(noise[N*i+j] <= noise_cut){
    if((i>1 && i<N-2) && (j>1 && j<N-2)){
      dL[N*i+j] = 20 * I[N*i+j].x -
                  8 * I[N*(i+1)+j].x - 8 * I[N*i+(j+1)].x - 8 * I[N*(i-1)+j].x - 8 * I[N*i+(j-1)].x +
                  2 * I[N*(i+1)+(j-1)].x + 2 * I[N*(i+1)+(j+1)].x + 2 * I[N*(i-1)+(j-1)].x + 2 * I[N*(i-1)+(j+1)].x +
                  I[N*(i+2)+j].x + I[N*i+(j+2)].x + I[N*(i-2)+j].x + I[N*i+(j-2)].x;

    }else{
      dL[N*i+j] = 0.0;
    }
  }

  dL[N*i+j] *= lambda;

}


__global__ void DChi2(float *noise, float *dChi2, cufftComplex *Vr, float *U, float *V, float *w, long N, long numVisibilities, float fg_scale, float noise_cut, float xobs, float yobs, float DELTAX, float DELTAY, float antenna_diameter, float pb_factor, float pb_cutoff, float freq)
{

	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  int x0 = xobs;
  int y0 = yobs;
  float x = (j - x0) * DELTAX * RPDEG;
  float y = (i - y0) * DELTAY * RPDEG;

	float Ukv, Vkv, cosk, sink, atten;

  atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, xobs, yobs, DELTAX, DELTAY);

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

__global__ void DChi2_total(float *dchi2_total, float *dchi2, long N)
{

	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  dchi2_total[N*i+j] += dchi2[N*i+j];
}

__global__ void DPhi(float *dphi, float *dchi2, float *dH, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  dphi[N*i+j] = dchi2[N*i+j] + dH[N*i+j];
}

__global__ void substraction(float *x, cufftComplex *xc, float *gc, float lambda, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  x[N*i+j] = xc[N*i+j].x - lambda*gc[N*i+j];
}

__global__ void projection(float *px, float *x, float MINPIX, long N){

  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;


  if(INFINITY < x[N*i+j]){
    px[N*i+j] = INFINITY;
  }else{
    px[N*i+j] = x[N*i+j];
  }

  if(MINPIX > px[N*i+j]){
    px[N*i+j] = MINPIX;
  }else{
    px[N*i+j] = px[N*i+j];
  }
}

__global__ void normVectorCalculation(float *normVector, float *gc, long N){
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  normVector[N*i+j] = gc[N*i+j] * gc[N*i+j];
}

__global__ void copyImage(cufftComplex *p, float *device_xt, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  p[N*i+j].x = device_xt[N*i+j];
}


__global__ void getDot_LBFGS_fComplex (float *aux_vector, cufftComplex *vec_1, float *vec_2, int k, int h, int M, int N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  aux_vector[N*i+j] = vec_1[M*N*k+N*i+j].x*vec_2[M*N*h+N*i+j];
}

__global__ void getDot_LBFGS_ff (float *aux_vector, float *vec_1, float *vec_2, int k, int h, int M, int N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  aux_vector[N*i+j] = vec_1[M*N*k+N*i+j]*vec_2[M*N*h+N*i+j];
}

__global__ void updateQ (float *d_q, float alpha, float *d_y, int k, int M, int N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  d_q[N*i+j] += alpha * d_y[M*N*k+N*i+j];
}

__global__ void updateQComplex (float *d_q, float alpha, cufftComplex *d_y, int k, int M, int N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  d_q[N*i+j] += alpha * d_y[M*N*k+N*i+j].x;
}

__global__ void getR (float *d_q, float scalar, int N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  d_q[N*i+j] = d_q[N*i+j] * scalar;
}

__global__ void calculateSandY (cufftComplex *d_s, float *d_y, cufftComplex *p, float *xi, cufftComplex *p_p, float *xi_p, int iter, int M, int N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  d_s[M*N*iter+N*i+j].x = p[N*i+j].x - p_p[N*i+j].x;
  d_y[M*N*iter+N*i+j] = xi[N*i+j]- (-1.0f*xi_p[N*i+j]);
}


__host__ float chiCuadrado(cufftComplex *I)
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
    clip<<<numBlocksNN, threadsPerBlockNN>>>(I, N, MINPIX);
    gpuErrchk(cudaDeviceSynchronize());
  }

  clipWNoise<<<numBlocksNN, threadsPerBlockNN>>>(device_fg_image, device_noise_image, I, N, noise_cut, MINPIX, eta);
  gpuErrchk(cudaDeviceSynchronize());


  if(iter>0 && lambda!=0.0){
    switch(reg_term){
      case 0:
        SVector<<<numBlocksNN, threadsPerBlockNN>>>(device_S, device_noise_image, device_fg_image, N, noise_cut, MINPIX, eta);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 1:
        QPVector<<<numBlocksNN, threadsPerBlockNN>>>(device_S, device_noise_image, device_fg_image, N, noise_cut);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 2:
        TVVector<<<numBlocksNN, threadsPerBlockNN>>>(device_S, device_noise_image, device_fg_image, N, noise_cut);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 3:
        LVector<<<numBlocksNN, threadsPerBlockNN>>>(device_S, device_noise_image, device_fg_image, N, noise_cut);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      default:
        printf("Selected prior is not defined\n");
        goToError();
        break;
    }
  }

  if(num_gpus == 1){
    cudaSetDevice(selected);
    for(int f=0; f<data.nfields;f++){
      for(int i=0; i<data.total_frequencies;i++){
        for(int sto=0; sto<data.nstokes; sto++){
          if(fields[f].numVisibilitiesPerFreqPerStoke[i][sto] > 0){
            gpuErrchk(cudaMemset(device_chi2, 0, sizeof(float)*data.max_number_visibilities_in_channel_and_stokes));

          	apply_beam<<<numBlocksNN, threadsPerBlockNN>>>(antenna_diameter, pb_factor, pb_cutoff, device_fg_image, device_image, N, fields[f].global_xobs, fields[f].global_yobs, fg_scale, fields[f].nu[i], DELTAX, DELTAY);
          	gpuErrchk(cudaDeviceSynchronize());

          	//FFT 2D
          	if ((cufftExecC2C(plan1GPU, (cufftComplex*)device_image, (cufftComplex*)device_V, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
          		printf("CUFFT exec error\n");
          		goToError();
          	}
          	gpuErrchk(cudaDeviceSynchronize());

            //PHASE_ROTATE
            phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(device_V, M, N, fields[f].global_xobs, fields[f].global_yobs);
          	gpuErrchk(cudaDeviceSynchronize());

            //RESIDUAL CALCULATION
            //if(!gridding){
            vis_mod<<<fields[f].visibilities[i][sto].numBlocksUV, fields[f].visibilities[i][sto].threadsPerBlockUV>>>(fields[f].device_visibilities[i][sto].Vm, device_V, fields[f].device_visibilities[i][sto].u, fields[f].device_visibilities[i][sto].v, fields[f].device_visibilities[i][sto].weight, deltau, deltav, fields[f].numVisibilitiesPerFreqPerStoke[i][sto], N);
          	gpuErrchk(cudaDeviceSynchronize());

            residual<<<fields[f].visibilities[i][sto].numBlocksUV, fields[f].visibilities[i][sto].threadsPerBlockUV>>>(fields[f].device_visibilities[i][sto].Vr, fields[f].device_visibilities[i][sto].Vm, fields[f].device_visibilities[i][sto].Vo, fields[f].numVisibilitiesPerFreqPerStoke[i][sto]);
            gpuErrchk(cudaDeviceSynchronize());
            /*}else{
              residual<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vr, device_V, fields[f].device_visibilities[i].Vo, fields[f].numVisibilitiesPerFreq[i]);
              gpuErrchk(cudaDeviceSynchronize());

            }*/

          	////chi 2 VECTOR
          	chi2Vector<<<fields[f].visibilities[i][sto].numBlocksUV, fields[f].visibilities[i][sto].threadsPerBlockUV>>>(device_chi2, fields[f].device_visibilities[i][sto].Vr, fields[f].device_visibilities[i][sto].weight, fields[f].numVisibilitiesPerFreqPerStoke[i][sto]);
          	gpuErrchk(cudaDeviceSynchronize());

          	//REDUCTIONS
          	//chi2
          	resultchi2  += deviceReduce<float>(device_chi2, fields[f].numVisibilitiesPerFreqPerStoke[i][sto]);
          }
        }
      }
    }
  }else{
    for(int f=0; f<data.nfields; f++){
      #pragma omp parallel for schedule(static,1)
      for (int i = 0; i < data.total_frequencies; i++)
  		{
        float result = 0.0;
        unsigned int j = omp_get_thread_num();
  			//unsigned int num_cpu_threads = omp_get_num_threads();
  			// set and check the CUDA device for this CPU thread
  			int gpu_id = -1;
  			cudaSetDevice((i%num_gpus) + firstgpu);   // "% num_gpus" allows more CPU threads than GPU devices
  			cudaGetDevice(&gpu_id);
        for(int sto=0; sto<data.nstokes; sto++){
          if(fields[f].numVisibilitiesPerFreqPerStoke[i][sto] > 0){

            gpuErrchk(cudaMemset(vars_gpu[i%num_gpus].device_chi2, 0, sizeof(float)*data.max_number_visibilities_in_channel_and_stokes));

          	apply_beam<<<numBlocksNN, threadsPerBlockNN>>>(antenna_diameter, pb_factor, pb_cutoff, vars_gpu[i%num_gpus].device_image, device_fg_image, N, fields[f].global_xobs, fields[f].global_yobs, fg_scale, fields[f].nu[i], DELTAX, DELTAY);
          	gpuErrchk(cudaDeviceSynchronize());

          	//FFT 2D
          	if ((cufftExecC2C(vars_gpu[i%num_gpus].plan, (cufftComplex*)vars_gpu[i%num_gpus].device_image, (cufftComplex*)vars_gpu[i%num_gpus].device_V, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
          		printf("CUFFT exec error\n");
          		//return -1 ;
          		goToError();
          	}
          	gpuErrchk(cudaDeviceSynchronize());

            //PHASE_ROTATE
            phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(vars_gpu[i%num_gpus].device_V, M, N, fields[f].global_xobs, fields[f].global_yobs);
          	gpuErrchk(cudaDeviceSynchronize());

            //RESIDUAL CALCULATION
            //if(!gridding){
            vis_mod<<<fields[f].visibilities[i][sto].numBlocksUV, fields[f].visibilities[i][sto].threadsPerBlockUV>>>(fields[f].device_visibilities[i][sto].Vm, vars_gpu[i%num_gpus].device_V, fields[f].device_visibilities[i][sto].u, fields[f].device_visibilities[i][sto].v, fields[f].device_visibilities[i][sto].weight, deltau, deltav, fields[f].numVisibilitiesPerFreqPerStoke[i][sto], N);
          	gpuErrchk(cudaDeviceSynchronize());

            residual<<<fields[f].visibilities[i][sto].numBlocksUV, fields[f].visibilities[i][sto].threadsPerBlockUV>>>(fields[f].device_visibilities[i][sto].Vr, fields[f].device_visibilities[i][sto].Vm, fields[f].device_visibilities[i][sto].Vo, fields[f].numVisibilitiesPerFreqPerStoke[i][sto]);
            gpuErrchk(cudaDeviceSynchronize());
          /*  }else{
              residual<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vr, vars_gpu[i%num_gpus].device_V, fields[f].device_visibilities[i].Vo, fields[f].numVisibilitiesPerFreq[i]);
              gpuErrchk(cudaDeviceSynchronize());
            }*/

          	////chi2 VECTOR
          	chi2Vector<<<fields[f].visibilities[i][sto].numBlocksUV, fields[f].visibilities[i][sto].threadsPerBlockUV>>>(vars_gpu[i%num_gpus].device_chi2, fields[f].device_visibilities[i][sto].Vr, fields[f].device_visibilities[i][sto].weight, fields[f].numVisibilitiesPerFreqPerStoke[i][sto]);
          	gpuErrchk(cudaDeviceSynchronize());


            result = deviceReduce<float>(vars_gpu[i%num_gpus].device_chi2, fields[f].numVisibilitiesPerFreqPerStoke[i][sto]);
          	//REDUCTIONS
          	//chi2
            #pragma omp critical
            {
              resultchi2  += result;
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
  resultS  = deviceReduce<float>(device_S, M*N);
  resultPhi = (0.5 * resultchi2) + (lambda * resultS);

  final_chi2 = resultchi2;
  final_S = resultS;
  /*printf("chi2 value = %.5f\n", resultchi2);
  printf("S value = %.5f\n", resultS);
  printf("(1/2) * chi2 value = %.5f\n", 0.5*resultchi2);
  printf("lambda * S value = %.5f\n", lambda*resultS);
  printf("Phi value = %.5f\n\n", resultPhi);*/

  return resultPhi;
}



__host__ void dchiCuadrado(cufftComplex *I, float *dxi2)
{

  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
    cudaSetDevice(firstgpu);
  }

  if(clip_flag){
    clip<<<numBlocksNN, threadsPerBlockNN>>>(I, N, MINPIX);
    gpuErrchk(cudaDeviceSynchronize());
  }

  gpuErrchk(cudaMemset(device_dchi2, 0, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_dphi, 0, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_dchi2_total, 0, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_dS, 0, sizeof(float)*M*N));

  if(print_images)
    fitsOutputCufftComplex(I, mod_in, out_image, mempath, iter, fg_scale, M, N, 1);

  if(iter>0 && lambda!=0.0){
    switch(reg_term){
      case 0:
        DS<<<numBlocksNN, threadsPerBlockNN>>>(device_dS, I, device_noise_image, noise_cut, lambda, MINPIX, eta, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 1:
        DQ<<<numBlocksNN, threadsPerBlockNN>>>(device_dS, I, device_noise_image, noise_cut, lambda, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 2:
        DTV<<<numBlocksNN, threadsPerBlockNN>>>(device_dS, I, device_noise_image, noise_cut, lambda, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      case 3:
        DL<<<numBlocksNN, threadsPerBlockNN>>>(device_dS, I, device_noise_image, noise_cut, lambda, N);
        gpuErrchk(cudaDeviceSynchronize());
        break;
      default:
        printf("Selected prior is not defined\n");
        goToError();
        break;
    }
  }

  if(num_gpus == 1){
    cudaSetDevice(selected);
    for(int f=0; f<data.nfields; f++){
      for(int i=0; i<data.total_frequencies;i++){
        for(int sto=0; sto<data.nstokes; sto++){
          if(fields[f].numVisibilitiesPerFreqPerStoke[i][sto] > 0){

            DChi2<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_dchi2, fields[f].device_visibilities[i][sto].Vr, fields[f].device_visibilities[i][sto].u, fields[f].device_visibilities[i][sto].v, fields[f].device_visibilities[i][sto].weight, N, fields[f].numVisibilitiesPerFreqPerStoke[i][sto], fg_scale, noise_cut, fields[f].global_xobs, fields[f].global_yobs, DELTAX, DELTAY, antenna_diameter, pb_factor, pb_cutoff, fields[f].nu[i]);
            gpuErrchk(cudaDeviceSynchronize());

            DChi2_total<<<numBlocksNN, threadsPerBlockNN>>>(device_dchi2_total, device_dchi2, N);
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
        for(int sto=0; sto<data.nstokes; sto++){
          if(fields[f].numVisibilitiesPerFreqPerStoke[i][sto] > 0){

            DChi2<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, vars_gpu[i%num_gpus].device_dchi2, fields[f].device_visibilities[i][sto].Vr, fields[f].device_visibilities[i][sto].u, fields[f].device_visibilities[i][sto].v, fields[f].device_visibilities[i][sto].weight, N, fields[f].numVisibilitiesPerFreqPerStoke[i][sto], fg_scale, noise_cut, fields[f].global_xobs, fields[f].global_yobs, DELTAX, DELTAY, antenna_diameter, pb_factor, pb_cutoff, fields[f].nu[i]);
            gpuErrchk(cudaDeviceSynchronize());


            #pragma omp critical
            {
              DChi2_total<<<numBlocksNN, threadsPerBlockNN>>>(device_dchi2_total, vars_gpu[i%num_gpus].device_dchi2, N);
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

  DPhi<<<numBlocksNN, threadsPerBlockNN>>>(device_dphi, device_dchi2_total, device_dS, N);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(dxi2, device_dphi, sizeof(float)*M*N, cudaMemcpyDeviceToDevice));
}
