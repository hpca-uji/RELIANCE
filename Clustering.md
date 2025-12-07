# PadChest clustering

We begin by describing the clustering applied to the PadChest dataset. In this hierarchy, labels from our proposed classification system are shown in **Bold** and capitalized, while original PadChest labels appear in lowercase. Note that not all PadChest labels are included in a cluster, and each PadChest label appears only once in the hierarchy — it is not duplicated across multiple levels. For instance, the PadChest label cardiomegaly is listed only under the **Cardiomegaly** label and does not reappear under broader categories like **MH** or **MH-urgent**.

**Mass**: pleural mass, mediastinal mass, mass, breast mass, pulmonary mass, soft tissue mass.

**Lung**: chronic changes, calcified granuloma, granuloma, cyst, cavitation, fibrotic band, volume loss, hypoexpansion, hypoexpansion basal, bullas, lung vascular paucity, air trapping, bronchiectasis, bronchovascular markings, increased density, azygos lobe, vascular redistribution, central vascular redistribution, kerley lines, tuberculosis, tuberculosis sequelae, lung metastasis, pulmonary fibrosis, post radiotherapy changes, asbestosis signs, heart insufficiency, respiratory distress, pulmonary hypertension, pulmonary artery hypertension, pulmonary venous hypertension, pulmonary edema, bone metastasis.

**Lung-urgent**: pulmonary mass.

**Nodules / Multiple nodules**: nodule, multiple nodules.

**Pneumonia**: pneumonia, atypical pneumonia.

**Infiltrates**: infiltrates, interstitial pattern, ground glass pattern, reticular interstitial pattern, reticulonodular interstitial pattern, miliary opacities, alveolar pattern, consolidation, air bronchogram.

**Atelectasis**: atelectasis, total atelectasis, lobar atelectasis, segmental atelectasis, laminar atelectasis, round atelectasis, atelectasis basal.

**COPD / Emphysema**: emphysema, COPD signs.

**MH**: calcified adenopathy, calcified mediastinal adenopathy, heart valve calcified, pneumomediastinum, mediastinal shift, dextrocardia, right sided aortic arch, aortic atheromatosis, aortic elongation, descendent aortic elongation, ascendent aortic elongation, aortic button enlargement, supra aortic elongation, aortic aneurysm, mediastinal enlargement, goiter, tracheal shift, esophagic dilatation, azygoesophageal recess shift, mediastinic lipomatosis.

**MH-urgent**: hilar congestion, cardiomegaly, superior mediastinal enlargement, mediastinal mass, tracheal shift.

**Cardiomegaly**: cardiomegaly.

**Hila**: hilar enlargement, adenopathy, hilar congestion.

**Vascular hilar enlargement**: vascular hilar enlargement, pulmonary artery enlargement.

**Aortic elongation**: aortic elongation, descendent aortic elongation, ascendent aortic elongation, aortic button enlargement, supra aortic elongation.

**PDTW**: calcified pleural plaques, pneumoperitoneo, flattened diaphragm, fissure thickening, minor fissure thickening, major fissure thickening, pleural plaques, pericardial effusion, thoracic cage deformation, scoliosis, kyphosis, pectum excavatum, pectum carinatum, cervical rib, blastic bone lesion, costochondral junction hypertrophy, sternoclavicular junction hypertrophy, axial hyperostosis, osteopenia, osteoporosis, subacromial space narrowing, gynecomastia, Chilaiditi sign, hemidiaphragm elevation, diaphragmatic eventration.

**PDTW-Urgent**: subcutaneous emphysema, pleural mass, lytic bone lesion, sclerotic bone lesion, blastic bone lesion, fracture, clavicle fracture, humeral fracture, vertebral fracture, rib fracture, soft tissue mass.

**Pleural effusion**: hydropneumothorax, loculated fissural effusion, pleural effusion, loculated pleural effusion.

**Pneumothorax**: pneumothorax, hydropneumothorax.

**Costophrenic angle blunting**: costophrenic angle blunting.

**PDTW-Incidental**: calcified pleural thickening, pleural thickening, apical pleural thickening, non axial articular degenerative changes, callus rib fracture.

**Vertebral degenerative changes**: vertebral degenerative changes.

**Hiatal hernia**: hiatal hernia.

**Foreign bodies**: artificial heart valve, artificial mitral heart valve, artificial aortic heart valve.

**Electrical devices**: electrical device, dual chamber device, single chamber device, pacemaker, dai.

**Tube**: tracheostomy tube, endotracheal tube, chest drain tube, ventriculoperitoneal drain tube.

**NSG Tube**: NSG tube.

**Catheter**: catheter, central venous catheter, central venous catheter via subclavian vein, central venous catheter via jugular vein, reservoir central venous catheter, central venous catheter via umbilical vein.

**Surgery**: surgery, metal, osteosynthesis material, sternotomy, suture material, bone cement, prosthesis, humeral prosthesis, mammary prosthesis, endoprosthesis, aortic endoprosthesis, surgery breast, mastectomy, surgery neck, surgery lung, surgery heart, surgery humeral.

**Calcification**: calcified densities, calcified granuloma, calcified adenopathy, calcified mediastinal adenopathy, calcified pleural plaques, heart valve calcified, calcified fibroadenoma, calcified pleural thickening, pleural plaques, aortic atheromatosis.

**Pseudonodule**: pseudonodule, nipple shadow, end on vessel.

**Suboptimal**: suboptimal study.

# ChestX-ray14 clustering

Next, we present the clustering applied to the ChestX-ray14 dataset. As before, labels from our proposed classification system are displayed in **Bold** and capitalized, while original ChestX-ray14 labels appear in lowercase. Each ChestX-ray14 label is assigned to a single cluster — specifically, to the most specific (i.e., lowest-level) applicable class in our hierarchy. Labels do not appear multiple times across different levels.

**Mass**: mass.

**Lung-urgent**: fibrosis, edema.

**Nodules**: nodule.

**Pneumonia**: pneumonia.

**Infiltrates**: infiltration, consolidation.

**Atelectasis**: atelectasis.

**COPD / Emphysema**: emphysema.

**Cardiomegaly**: cardiomegaly.

**Pleural effusion**: effusion.

**Pneumothorax**: pneumothorax.

**PDTW-incidental**: pleural thickening.

**Hiatal hernia**: hernia.