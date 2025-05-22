# LoVoCCS-BCG-Ident
This repository contains everything related to the manual identification of LoVoCCS galaxy cluster BCG candidates and their spectra/spectroscopic redshifts.

We make use of a prototype framework for simplifying and speeding up the manual identification of BCG candidates (we believe that this is still the most effective method) - https://github.com/DavidT3/BCG-Ident-Framework
The general steps are as follows:
1. DESI Legacy Survey DR10 optical/NIR and XMM X-ray images are viewed and interacted with simultaneously, in order to identify BCG candidates associated with our clusters.
2. The same Legacy Survey photometry is now used in a spectrum identification tool. The tool fetches optical/NIR galaxy spectra around the cluster from DESI DR1 and RCSEDv2 (this has SDSS/eBOSS, GAMA, Hectospec, 6dF, CfA, FAST, and other spectra). We interactively identify spectra associated with our candidates.
3. Identifying information for each spectrum associated with a candidate is stored, and the spectroscopic redshifts are read out and saved.

## Outputs

* File containing the RA-Dec coordinates of all BCG candidates visually identified for each LoVoCCS cluster; note that these are ***candidates***, and you should exercise due care. Some BCG candidates may also be more related to member galaxy & X-ray substructure that might constitute an infalling group - ***[outputs/bcg_output_sample.csv]***

* Fiducial redshift tables for the BCG candidates - all entries are spectroscopic redshifts, and are taken from the tool-identified spectra and literature sources (in cases where no tool identified spectrum was found). In cases where multiple spectra were identified, we have attempted to select the 'best' one as the source of the redshift. - ***[outputs/fiducial_cand_redshift_tables/BCG\<CANDNO\>_fiducial_specz.csv]***

* Files containing the results of cross-matches of our BCG candidates with VLASS and GALEX source catalogs. There may be multiple entries per particular cluster BCG candidate, as we didn't really perform an actual _match_, more of a search within a specified radius. - ***[outputs/vlass_galex_crossmatches/bcg\<CANDNO\>_cands_\<CATALOG\>_searchrad\<SEARCH RADIUS\>arcsec.csv]***

* History of the use of the BCG identification framework, including all image file paths, identified BCG candidate positions, source/unique ID information for spectra assigned to candidates, and redshift information - ***[history/bcg_ident_proj_save.json]***

* Spectroscopic redshifts identified from literature for candidates with no tool spectra. Files also include the source dataset, and any notes - ***[outputs/literature_specz_for_cands_without/bcg\<CANDNO\>_cand_notoolz_litspecz.json]***

* Notes on the manual search of tool-identified candidate spectra for significant emission lines - ***[outputs/rcsedv2_desidr1_spec_notes/bcg\<CANDNO\>_emline_notes.json]***

* Which candidates were found to have ESO-served spectral cubes (i.e. MUSE, ALMA, and KMOS) - ***[outputs/eso_cube_notes/bcg\<CANDNO\>_eso_cube_notes.json]***

* Which candidates were found to have MANGA spectral cubes - ***[outputs/manga_cube_notes/bcg\<CANDNO\>_manga_cube_notes.json]***

* Simple visualizations of tool-identified spectra for each candidate, note that some tool-identified spectra only have redshift information and no raw data, so visualisations may be missing. - ***[indiv_cluster/\<CLUSTER NAME\>/\<CLUSTER NAME\>_BCG\<CANDNO\>.pdf]***

* Simple visualizations manually identified BCG candidate positions, and LoVoCCS-II BCG positions, overplotted on DESI Legacy Survey and XMM images. - ***[indiv_cluster/\<CLUSTER NAME\>/\<CLUSTER NAME\>_LoVoCCSII-BCG_manual_BCGcands_comparison.png]***

## Notebook summaries

* ***STEP1-spot_the_bcg.ipynb*** - First step of the BCG identification framework, displays interactive DESI Legacy Survey and XMM images, and allows users to click on galaxies and designate that position as a BCG candidate.
* ***STEP2-search_for_spectra.ipynb*** - Follows on from the initial step, fetches spectra from a variety of datasets, and displays their positions on Legacy Survey images, the user can then click on a spectrum to load a visualisation, and designate selected spectra as being associated with the current candidate.
* ***STEP3-compare_bcg_redshifts_to_cluster.ipynb*** - Here the BCG candidate spectroscopic redshifts are compared to original MCXC redshifts - there are several sets of figures showing the comparisons. Candidates without associated spectra are identified.
* ***STEP4-fiducial_redshift_table.ipynb*** - This notebook is used to assemble a set of 'fiducial redshift' tables for the various BCG candidates - it attempts to choose between spectroscopic redshifts when multiple spectra have been found, and folds in the information collected from manual identification of missing redshifts from literature.

* ***manual_bcgs-vs-LoVoCCSII_bcgs.ipynb*** - Puts the manually identified BCGs into context by comparing their positions to those of the BCGs identified for the LoVoCCS-II paper - position indicators are plotted on DESI Legacy Survey and XMM images.
*  ***crossmatch_bcgcands_VLASS_GALEX.ipynb*** - Uses the AstroQuery interface with VizieR to cross match our manually identified BCG candidates with the VLASS QL Ep.1 and GALEX AIS UV GR6+7 source catalogs - really we are searching for catalog entries within search radii (can be changed easily for both catalogs), as we do not only take the closest source (this is more important for VLASS than GALEX, as big jets and wide-angle-tail type structures can often be split into muliple sources in the catalog).
