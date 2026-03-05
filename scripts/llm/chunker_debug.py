from scripts.llm.llm_module import *


chunker = TextChunker(method = "semantic", embedding_model_name = "BAAI/bge-small-en-v1.5")
text2 = '''
 
Name:  ___             Unit No:   ___
 
Admission Date:  ___              Discharge Date:   ___
 
Date of Birth:  ___             Sex:   M
 
Service: MEDICINE
 
Allergies: 
MS ___ / Enbrel / infliximab
 
Attending: ___
 
Chief Complaint:
Fever, joint pain, diarrhea
 
Major Surgical or Invasive Procedure:
None

 
Past Medical History:
- Ankylosing spondylitis
- Ulcerative colitis s/p total proctocolectomy
- Syncope
- Sleep apnea
- Narcotic abuse w/ Narcotics agreement (with Dr. ___
- DVT
- Suicide attempt (___)
- Depression
- LLL segmental/subsegmental PE (___)
- Paroxysmal atrial fibrillation
- Steroid-induced diabetes mellitus
 
Social History:
___
Family History:
- Mother with rheumatoid arthritis
- Maternal aunt with lupus
- Heart disease on father's side 
- Uncle with MI
 
Physical Exam:
Admission Physical exam
=======================
VITALS: ___ Temp: 97.8 PO BP: 127/74 HR: 72 RR: 16 O2
sat: 96% 
GENERAL: Alert and interactive. Pleasant and conversant. 
HEENT: MMM.
CARDIAC: Regular rhythm, normal rate. Audible S1 and S2. No
murmurs/rubs/gallops.
LUNGS: Clear to auscultation bilaterally. No wheezes, rhonchi or
rales. No increased work of breathing.
ABDOMEN: Normal bowel sounds. Non-distended. Tender to palpation
in lower quadrants bilaterally.
EXTREMITIES: Lateral hips tender to palpation bilaterally. L 
knee
mildly swollen and tender to palpation. ___ and ___ MCP joints
tender to palpation bilaterally. No edema. Pulses DP/Radial 2+
bilaterally.
SKIN: Warm and perfused. Cap refill <2s. No rashes.

Discharge Physical exam
=======================
VITAL SIGNS: Temp: 97.6 PO BP: 117/74 HR: 75 RR: 18 O2 sat: 98%
O2 delivery: Ra FSBG: 96 
CV: RRR. No murmurs, rubs, or gallops. Normal S1 S2.
PULM: CTAB. No increased work of breathing.
ABD: Soft and non-distended. Mild tenderness to palpation in the
lower abdomen.
EXT: Lateral hips tender to palpation bilaterally. Improved from
prior. L knee with some fluid but no erythema. Mildly warm. No
edema.
SKIN: Warm and well perfused.

 
Pertinent Results:
Labs
====

___ 10:44PM BLOOD WBC-11.6* RBC-4.44* Hgb-13.1* Hct-38.6* 
MCV-87 MCH-29.5 MCHC-33.9 RDW-14.3 RDWSD-44.1 Plt ___
___ 10:44PM BLOOD Neuts-48.7 ___ Monos-10.1 Eos-1.6 
Baso-0.9 Im ___ AbsNeut-5.64 AbsLymp-4.34* AbsMono-1.17* 
AbsEos-0.18 AbsBaso-0.11*
___ 10:44PM BLOOD Plt ___
___ 10:44PM BLOOD Glucose-97 UreaN-10 Creat-0.8 Na-143 
K-3.8 Cl-103 HCO3-26 AnGap-14
___ 10:44PM BLOOD ALT-29 AST-21 AlkPhos-145* TotBili-0.2
___ 10:44PM BLOOD Lipase-33
___ 10:44PM BLOOD Albumin-4.2 Calcium-9.7 Phos-5.0* Mg-1.9
___ 10:44PM BLOOD CRP-2.8
___ 02:06AM BLOOD Lactate-1.5
___ 05:33AM BLOOD Glucose-165* UreaN-13 Creat-0.7 Na-138 
K-4.9 Cl-97 HCO3-26 AnGap-15
___ 05:30AM BLOOD WBC-13.4* RBC-4.48* Hgb-13.4* Hct-40.3 
MCV-90 MCH-29.9 MCHC-33.3 RDW-15.3 RDWSD-50.4* Plt ___
___ 01:54AM URINE Blood-NEG Nitrite-NEG Protein-30* 
Glucose-NEG Ketone-TR* Bilirub-NEG Urobiln-2* pH-6.0 Leuks-SM*

Micro
=====
___ 1:35 am BLOOD CULTURE

                            **FINAL REPORT ___

   Blood Culture, Routine (Final ___:    NO GROWTH. 

 
 
___ 1:54 am URINE

                            **FINAL REPORT ___

   URINE CULTURE (Final ___: 
      MIXED BACTERIAL FLORA ( >= 3 COLONY TYPES), CONSISTENT 
WITH SKIN
      AND/OR GENITAL CONTAMINATION. 

 
 
___ 2:55 pm STOOL     CONSISTENCY: NOT APPLICABLE
      Source: Stool. 

                            **FINAL REPORT ___

   OVA + PARASITES (Final ___: 
      NO OVA AND PARASITES SEEN. 
      This test does not reliably detect Cryptosporidium, 
Cyclospora or
      Microsporidium. While most cases of Giardia are detected 
by routine
      O+P, the Giardia antigen test may enhance detection when 
organisms
      are rare. 

Imaging
=======
ABD Xray ___

IMPRESSION:  
  
Nonspecific, nonobstructive bowel gas pattern. 

Gastric emptying study - ___

IMPRESSION:  Irregular gastric emptying with a prolonged 
half-emptying time. 
The findings suggest abnormally delayed gastric emptying; 
however, normal values 
are generated in an outpatient setting and application of normal 
values to 
acutely ill patients should be made with caution. 
  
CXR ___

IMPRESSION:  
  
Comparison to ___.  New subtle parenchymal opacity at 
the bases of 
the right lung, with air bronchograms, and reflecting pneumonia 
in the 
appropriate clinical setting.  No other changes are noted.  
Normal size of the 
heart.  No pleural effusions.  No pulmonary edema.  No 
pneumothorax.  Normal 
hilar and mediastinal contours. 

Barrium Swallow ___

IMPRESSION: 
  
1. Abnormal esophageal motility with the barium tablet being 
held up in the 
mid esophagus.  The barium tablet did not clear with multiple 
water and thin 
barium swallows. 
2. No strictures or esophageal mass identified at this level. 
3. Mild delayed esophageal motility with liquid contrast, but no 
gross liquid 
held up as demonstrated with the barium tablet. 
4. No hiatal hernia. 

  

 
Brief Hospital Course:
___ w/ history of ulcerative colitis s/p proctocolectomy and 
ankylosing spondylitis on chronic opioids, treated for suspected 
CAP, now with prolonged PO intolerance of unclear etiology 
amidst AS flare.

TRANSITIONAL ISSUES:
====================
[ ] ___ was discharged with short course of narcotics 
(oxycodone and oxycontin) for pain management. We recommend 
outpatient followup with his PCP for ongoing management
 [ ] Newly started on long-acting OxyCODONE SR (OxyCONTIN) 10 mg 
PO BID for pain, PCP ___ be prescribing and managing your pain 
medications going forward 
 [ ] Changed dose of OxyCODONE (Immediate Release) 5 mg PO/NG 
Q6H:PRN for breakthrough pain
[ ] An additional medication, lorazepam, was added to help the 
___ tolerate food in the setting of frequent emesis. This 
medication may interact with the ___ current pain 
medications. We recommend that the ___ follow up with his 
PCP to discuss the current regimen.
[ ] Continue Prednisone as directed, until next appointment with 
Dr. ___: 
- 20mg daily on ___ daily afterward (___-), until rheumatology follow-up; 
final steroid taper to be determined 
[ ] Continue Certolizumab for AS; next dose ___ 
[ ] Needs repeat gastric emptying study and esophageal manometry 
as outpatient, gastroenterology follow-up arranged 

ACTIVE ISSUES:
==============
# Emesis / PO Intolerance
Unclear etiology. Initially was unable to tolerate pills or 
certain liquids for more than a few minutes. ___ denies 
dysphagia but says he experiences anxiety when swallowing 
because he anticipates vomiting. Denies nausea. Gastric emptying 
study showing delayed emptying. Delayed emptying possibly 
related to chronic opioid use and steroid-induced DM. 
Metoclopramide trialed twice, second time given with Dilaudid, 
however stopped due to persistent paresthesias and jitteriness. 
Tolerating PO meals with 0.5mg lorazepam. PCP is OK with 
prescribing opioid/lorazepam combination. Barium swallow ___ 
showed abnormal esophageal motility with the barium tablet held 
up in the mid-esophagus. No strictures or esophageal mass. GI 
recommending education and outpatient manometry, as well as 
repeat gastric emptying as an outpatient. 

# AS Flare
# Pain Management
Followed by Dr. ___ at ___ has been on multiple DMARDs 
however eventually became intolerant to each for various 
reasons. Most recently did not tolerate Cosentyx due to 
neuropathy (last dose ___. ___ reports polyarthritis was 
previously well-controlled with DMARDs and ibuprofen. Presented 
with bilateral hip and knee pain. Despite normal ESR/CRP, most 
likely an episode of AS flare given cessation of most recent 
DMARD. Started on ___ IV methylprednisolone with minimal 
improvement. Received first dose of Cimzia ___. Unclear if 
opioid pain regimen is significantly alleviating joint pain. 
Stopped continuous O2 monitoring given stable O2 sat over past 
several nights. Currently on PO steroid taper per rheum recs. 
Started on long-acting oxycodone in addition to continuation of 
short-acting formulation; pain regimen OxyCONTIN 10 mg PO BID 
with Oxycodone 5 mg PO Q6H:PRN for breakthrough.

# Diarrhea
Presented with acute on subacute chronic watery diarrhea. Low 
suspicion for UC flare given normal inflammatory markers. 
Osmotic unlikely given ___ has not had significant PO 
intake. Re-started PR mesalamine as per GI did not think his 
erosions from previous pouchoscopy were due to medication. 
___ tolerating PO loperamide with little effect. Stool O&P 
negative. ___ received Mesalamine 1000 mg PR QD. GI 
follow-up scheduled. 

# Leukocytosis
WBC wnl during admission. Elevated to 14.5 on ___. Repeat 9.3. 
Elevated again to 13.5 on ___ ___. Remained elevated to 13.4 on 
___. ___ is afebrile but reports increased pain in hands and 
swelling and warmth in left knee. Denies cough or shortness of 
breath. Treated this admission for RLL CAP w/ 5-day course of 
levofloxacin. Low threshold for infectious workup given ___ 
has long history of immune suppression. CXR negative for 
consolidation. Low likelihood of pneumonia. Blood cultures 
negative for 24hrs.

CHRONIC/STABLE ISSUES:
======================
# Hx DVT and PE: Home rivaroxaban replaced with SQH given PO 
intolerance. Home rivaroxaban restarted when ___ was 
tolerant of PO medication.

# Testosterone replacement therapy: Missed initial dose on ___. 
Continued home testosterone cypionate with doses on ___. 
Medication to be administered on ___ going forward.

# Neuropathy: Continued home gabapentin 

# BPH: Continued home tamsulosin, finasteride 

# GERD: Continued home pantoprazole

=====

# CODE: Full (presumed)
# CONTACT: Proxy: ___ (sister) ___ 

This ___ was prescribed, or continued on, an opioid pain 
medication at the time of discharge (please see the attached 
medication list for details). As part of our safe opioid 
prescribing process, all patients are provided with an opioid 
risks and treatment resource education sheet and encouraged to 
discuss this therapy with their outpatient providers to 
determine if opioid pain medication is still indicated. 
 
Medications on Admission:
The Preadmission Medication list is accurate and complete.
1. Ascorbic Acid ___ mg PO DAILY 
2. Ferrous Sulfate 325 mg PO DAILY 
3. Finasteride 5 mg PO DAILY 
4. Gabapentin 600 mg PO QAM AND NOON  
5. Gabapentin 900 mg PO QHS 
6. Hydroxychloroquine Sulfate 200 mg PO BID 
7. LOPERamide 4 mg PO TID diarrhea 
8. OxyCODONE (Immediate Release) 7.5 mg PO Q4H:PRN Pain - 
Moderate 
9. Pantoprazole 40 mg PO Q12H 
10. PredniSONE 5 mg PO DAILY 
11. Rivaroxaban 20 mg PO DAILY 
12. Tamsulosin 0.4 mg PO QHS 
13. Vitamin D 400 UNIT PO DAILY 
14. Vitamin E 400 UNIT PO DAILY 
15. Acetaminophen 1000 mg PO Q8H 
16. Hydrocortisone Cream 1% 1 Appl TP BID ___ area 
17. Ondansetron 8 mg PO Q8H:PRN Nausea/Vomiting - First Line 
18. Testosterone Cypionate 200 mg IM 1X/WEEK (FR) 

 
Discharge Medications:
1.  Cimzia (certolizumab pegol) 400 mg/2 mL (200 mg/mL x 2) 
subcutaneous EVERY 2 WEEKS (FR) 
Take every 2 weeks. Next dose is ___  
2.  LORazepam 0.5 mg PO TID 
RX *lorazepam 0.5 mg 0.5 (One half) mg by mouth three times a 
day Disp #*24 Tablet Refills:*0 
3.  Mesalamine (Rectal) ___AILY 
RX *mesalamine 1,000 mg 1 suppository(s) rectally once a day 
Disp #*25 Suppository Refills:*0 
4.  OxyCODONE SR (OxyCONTIN) 10 mg PO BID 
RX *oxycodone [OxyContin] 10 mg 1 tablet(s) by mouth twice a day 
Disp #*16 Tablet Refills:*0 
5.  OxyCODONE (Immediate Release) 5 mg PO Q6H:PRN Pain - 
Moderate 
RX *oxycodone 5 mg 1 tablet(s) by mouth every six (6) hours Disp 
#*32 Capsule Refills:*0 
6.  PredniSONE 20 mg PO DAILY Duration: 3 Doses 
This is dose # 2 of 2 tapered doses
RX *prednisone 20 mg 1 tablet(s) by mouth Once Disp #*1 Tablet 
Refills:*0 
7.  PredniSONE 10 mg PO DAILY 
This is the maintenance dose to follow the last tapered dose
RX *prednisone 10 mg 1 tablet(s) by mouth once a day Disp #*20 
Tablet Refills:*0 
8.  Acetaminophen 1000 mg PO Q8H  
9.  Ascorbic Acid ___ mg PO DAILY  
10.  Ferrous Sulfate 325 mg PO DAILY  
11.  Finasteride 5 mg PO DAILY  
12.  Gabapentin 600 mg PO QAM AND NOON   
13.  Gabapentin 900 mg PO QHS  
14.  Hydrocortisone Cream 1% 1 Appl TP BID ___ area  
15.  Hydroxychloroquine Sulfate 200 mg PO BID  
16.  LOPERamide 4 mg PO TID diarrhea  
17.  Ondansetron 8 mg PO Q8H:PRN Nausea/Vomiting - First Line  
18.  Pantoprazole 40 mg PO Q12H  
19.  Rivaroxaban 20 mg PO DAILY  
20.  Tamsulosin 0.4 mg PO QHS  
21.  Testosterone Cypionate 200 mg IM 1X/WEEK (FR)  
22.  Vitamin D 400 UNIT PO DAILY  
23.  Vitamin E 400 UNIT PO DAILY  

 
Discharge Disposition:
Home
 
Discharge Diagnosis:
Primary diagnosis
=================
Ankylosing Spondylitis flare

Secondary Diagnosis
===================
Emesis / PO Intolerance
Diarrhea
DVT/PE

 
Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.
Activity Status: Ambulatory - Independent.

 
Discharge Instructions:
======================  
DISCHARGE INSTRUCTIONS  
======================  
Dear Mr. ___,  
It was a privilege caring for you at ___.  

WHY WAS I IN THE HOSPITAL?  
You came to the hospital because of vomiting, fevers, joint pain 
and worsening diarrhea.

WHAT HAPPENED TO ME IN THE HOSPITAL?  
In the hospital we increased your steroid dose and pain 
medications to treat your joint pain, which was being caused by 
a flare of your ankylosing spondylitis. Because you continued to 
have vomiting, we performed imaging to determine if your stomach 
was processing food at a normal speed. This imaging showed 
slight slowing of your stomach. We tried to give you medication 
that would speed up your system, however it made you feel 
jittery, so we stopped. We then gave you relaxation medications 
which improved your ability to tolerate food. You received a 
dose of your new biologic drug, Cimzia. You had difficulty 
swallowing your pain medications by mouth and experienced 
frequent vomiting. GI performed a study that showed a pill 
temporarily stuck in your esophagus but no other abnormalities. 
They recommended further work-up in the outpatient setting. We 
adjusted your pain medications to include a longer-acting drug 
that you were able to tolerate because you could take it with 
food.

NEW/CHANGED/STOPPED MEDICATIONS:
[ ] Continue Prednisone as directed, until next appointment with 
Dr. ___: 
- 20mg daily on ___ daily afterward (___-), until rheumatology follow-up 
[ ] Continue Certolizumab for AS; next dose ___ 
[ ] Newly started on long-acting OxyCODONE SR (OxyCONTIN) 10 mg 
PO BID for pain, Dr. ___ will be prescribing and managing your 
pain medications going forward 
[ ] Changed dose of OxyCODONE (Immediate Release) 5 mg PO/NG 
Q6H:PRN for breakthrough pain

WHAT SHOULD I DO AFTER I LEAVE THE HOSPITAL?  
-Please continue to take all of your medications and follow-up 
with your appointments as listed below.

We wish you the best!  

Sincerely,  
Your ___ Team  
 
Followup Instructions:
___
'''

ma = chunker.chunk_text(text2)
ma