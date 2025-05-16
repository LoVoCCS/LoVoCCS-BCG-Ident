# LoVoCCS-BCG-Ident
This repository contains everything related to the manual identification of LoVoCCS galaxy cluster BCG candidates and their spectra/spectroscopic redshifts.

We make use of a prototype framework for simplifying and speeding up the manual identification of BCG candidates (we believe that this is still the most effective method) - https://github.com/DavidT3/BCG-Ident-Framework
The general steps are as follows:
1. DESI Legacy Survey DR10 optical/NIR and XMM X-ray images are viewed and interacted with simultaneously, in order to identify BCG candidates associated with our clusters.
2. The same Legacy Survey photometry is now used in a spectrum identification tool. The tool fetches optical/NIR galaxy spectra around the cluster from DESI DR1 and RCSEDv2 (this has SDSS/eBOSS, GAMA, Hectospec, 6dF, CfA, FAST, and other spectra). We interactively identify spectra associated with our candidates.
3. Identifying information for each spectrum associated with a candidate is stored, and the spectroscopic redshifts are read out and saved.

## Outputs

* File containing the RA-Dec coordinates of all BCG candidates visually identified for each LoVoCCS cluster; note that these are ***candidates***, and you should exercise due care. Some BCG candidates may also be more related to member galaxy & X-ray substructure that might constitute an infalling group - ***[outputs/bcg_output_sample.csv]***
  
* <span style="color:red">**Add an output csv of all spec-zs**</span>

* History of the use of the BCG identification framework, including all image file paths, identified BCG candidate positions, source/unique ID information for spectra assigned to candidates, and redshift information - ***[history/bcg_ident_proj_save.json]***

* Spectroscopic redshifts identified from literature for candidates with no tool spectra. Files also include the source dataset, and any notes - ***[outputs/literature_specz_for_cands_without/bcg\<CANDNO\>_cand_notoolz_litspecz.json]***

* Notes on the manual search of tool-identified candidate spectra for significant emission lines - ***[outputs/rcsedv2_desidr1_spec_notes/bcg\<CANDNO\>_emline_notes.json]***

* Which candidates were found to have ESO-served spectral cubes (i.e. MUSE, ALMA, and KMOS) - ***[outputs/eso_cube_notes/bcg\<CANDNO\>_eso_cube_notes.json]***

* Which candidates were found to have MANGA spectral cubes - ***[outputs/manga_cube_notes/bcg\<CANDNO\>_manga_cube_notes.json]***

* Simple visualizations of tool-identified spectra for each candidate, note that some tool-identified spectra only have redshift information and no raw data, so visualisations may be missing - ***[indiv_cluster/\<CLUSTER NAME\>/\<CLUSTER NAME\>_BCG\<CANDNO\>.pdf]***

## Notebook summaries

*
