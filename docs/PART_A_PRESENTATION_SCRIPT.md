# **Part A – Student Presentation Script (អត្ថបទបទបង្ហាញ ផ្នែក A - ទ្វេភាសា អង់គ្លេស-ខ្មែរ)**

**Topic / ប្រធានបទ:** Student Record Prediction — Part A Data Analysis (ការទស្សន៍ទាយពិន្ទុសិស្ស — ផ្នែក A៖ ការវិភាគទិន្នន័យ)\
**Speaking time / រយៈពេលនិយាយ:** About 4–5 minutes at a comfortable pace (questions are separate) / ប្រហែល ៤-៥ នាទីក្នុងល្បឿនធម្មតា\
**Reference / ឯកសារយោង:** [Full Part A Data Analysis Guide](PART_A_DATA_ANALYSIS_GUIDE.md) | [កំណែជាភាសាខ្មែរសុទ្ធ](PART_A_PRESENTATION_SCRIPT_KH.md)

---

## **1. Define the problem (កំណត់បញ្ហា)**

**🗣️ English Script:**

> Good morning, teacher. Today I will explain **Part A: Data Analysis** of my Student Record Prediction project. My purpose is to understand academic scores, clean the data, and find useful patterns. In this presentation, I focus only on data analysis, not model training or prediction. The main value is **`final_score`**, the final examination score out of 100.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**

> ជម្រាបសួរលោកគ្រូ/អ្នកគ្រូ។ ថ្ងៃនេះខ្ញុំនឹងធ្វើបទបង្ហាញពន្យល់អំពី **ផ្នែក A៖ ការវិភាគទិន្នន័យ (Part A: Data Analysis)** នៃគម្រោងទស្សន៍ទាយកំណត់ត្រាសិស្សរបស់ខ្ញុំ។ គោលបំណងចម្បងរបស់ខ្ញុំ គឺដើម្បីស្វែងយល់ពីពិន្ទុសិក្សា សម្អាតទិន្នន័យ និងស្វែងរកគំរូទិន្នន័យ (patterns) ដែលមានប្រយោជន៍។ នៅក្នុងបទបង្ហាញនេះ ខ្ញុំផ្តោតតែលើការវិភាគទិន្នន័យប៉ុណ្ណោះ មិនទាន់ឈានដល់ការបង្វឹកម៉ូដែល (Model Training) ឬការទស្សន៍ទាយ (Prediction) ទេ។ អថេរគោលដៅសំខាន់បំផុតគឺ **`final_score`** ដែលជាពិន្ទុប្រឡងបញ្ចប់ឆមាសគិតលើពិន្ទុពេញ ១០០។

---

## **2. Collect a dataset (ប្រមូលសំណុំទិន្នន័យ)**

**🗣️ English Script:**

> My dataset is saved in [`data/raw/students_raw.csv`](../data/raw/students_raw.csv). It is a **synthetic academic benchmark**, generated for this project, not actual student records. It originally has **402 rows and 7 columns**: student ID, attendance, weekly study hours, assignment average, midterm score, previous GPA, and final score. The [data dictionary](../data/raw/DATA_DICTIONARY.md) explains each column.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**

> សំណុំទិន្នន័យរបស់ខ្ញុំត្រូវបានរក្សាទុកក្នុង [`data/raw/students_raw.csv`](../data/raw/students_raw.csv)។ វាជា **ទិន្នន័យគំរូសិប្បនិម្មិត (Synthetic academic benchmark)** ដែលបង្កើតឡើងសម្រាប់គម្រោងនេះ មិនមែនជាទិន្នន័យពិតរបស់សិស្សជាក់ស្តែងទេ។ ដើមឡើយវាមានទិន្នន័យចំនួន **៤០២ ជួរដេក និង ៧ ជួរឈរ**៖ អត្តសញ្ញាណសិស្ស (Student ID), វត្តមាន (attendance), ម៉ោងស្វ័យសិក្សាប្រចាំសប្តាហ៍ (weekly study hours), ពិន្ទុមធ្យមភាគកិច្ចការផ្ទះ (assignment average), ពិន្ទុពាក់កណ្តាលឆមាស (midterm score), និទ្ទេស GPA ពីមុន (previous GPA) និងពិន្ទុប្រឡងបញ្ចប់ (final score)។ ឯកសារ [Data Dictionary](../data/raw/DATA_DICTIONARY.md) ពន្យល់លម្អិតអំពីជួរឈរនីមួយៗ។

---

## 3. **Clean the data (សម្អាតទិន្នន័យ)**

**🗣️ English Script:**

> First, I checked for missing, duplicate, and invalid values. I removed **2 duplicate student IDs** and **2 records without final scores**. I also found one impossible attendance value of **125%** and marked it as missing so it could be corrected. I standardized student IDs and checked numerical ranges. After cleaning, I had **398 rows**. The steps are in my [cleaning notebook](../notebooks/01_data_cleaning.ipynb).

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**

> ដំបូង ខ្ញុំបានពិនិត្យរកមើលទិន្នន័យដែលបាត់ (missing), ជាន់គ្នា (duplicates) និងទិន្នន័យមិនត្រឹមត្រូវ (invalid values)។ ខ្ញុំបានលុប **ID សិស្សដែលជាន់គ្នា ចំនួន ២ ជួរ** និង **ទិន្នន័យដែលគ្មានពិន្ទុបញ្ចប់ ចំនួន ២ ជួរ**។ ខ្ញុំក៏បានរកឃើញតម្លៃវត្តមានមិនសមហេតុផលមួយគឺ **១២៥%** ហើយបានកំណត់វាជាតម្លៃបាត់ (missing) ដើម្បីកែតម្រូវ។ ខ្ញុំបានកែទម្រង់ Student ID ឱ្យមានស្តង់ដាររួម និងពិនិត្យដែនកំណត់នៃលេខ។ ក្រោយសម្អាតរួច សំណុំទិន្នន័យនៅសល់ **៣៩៨ ជួរ**។ ជំហានទាំងនេះមាននៅក្នុង [Cleaning Notebook](../notebooks/01_data_cleaning.ipynb)។

---

## **4. Handle missing values (ដោះស្រាយទិន្នន័យដែលបាត់បង់)**

**🗣️ English Script:**

> After those first cleaning steps, there were **1 missing attendance value, 8 study-hour values, 6 assignment averages, and 5 previous GPAs**. I used the **mean**, or average, for attendance, study hours, and GPA. I used the **median**, or middle value, for assignment scores because the distribution was more uneven. I did not invent missing final scores. The final dataset has **zero missing values**.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**

> ក្រោយពីជំហានសម្អាតដំបូង យើងនៅសល់ទិន្នន័យបាត់មួយចំនួនគឺ **វត្តមានបាត់ ១, ម៉ោងសិក្សាបាត់ ៨, ពិន្ទុកិច្ចការផ្ទះបាត់ ៦, និង GPA មុនៗបាត់ ៥**។ ខ្ញុំបានប្រើ **មធ្យមភាគ (Mean)** សម្រាប់វត្តមាន, ម៉ោងសិក្សា និង GPA។ ខ្ញុំបានប្រើ **មេដ្យាន (Median - តម្លៃកណ្តាល)** សម្រាប់ពិន្ទុកិច្ចការផ្ទះ ពីព្រោះការបែងចែកទិន្នន័យរបស់វាមានភាពលម្អៀង (uneven/skewed)។ ខ្ញុំមិនបានស្មាន ឬបំពេញពិន្ទុបញ្ចប់ (final score) ដោយខ្លួនឯងឡើយ។ សំណុំទិន្នន័យចុងក្រោយគឺ **គ្មានទិន្នន័យបាត់បង់ឡើយ (0 missing values)**។

---

## **5. Perform exploratory data analysis (រុករកនិងវិភាគទិន្នន័យ - EDA)**

**🗣️ English Script:**

> Next, I used [`notebooks/02_eda.ipynb`](../notebooks/02_eda.ipynb) for **EDA**, which means Exploratory Data Analysis. EDA helps us understand common values, unusual scores, and relationships between columns. I examined distributions, compared attendance groups, and checked correlations.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**

> បន្ទាប់មក ខ្ញុំបានប្រើ [`notebooks/02_eda.ipynb`](../notebooks/02_eda.ipynb) សម្រាប់ធ្វើ **EDA** (Exploratory Data Analysis)។ EDA ជួយយើងឱ្យយល់ដឹងអំពីតម្លៃទូទៅនៃទិន្នន័យ, ពិន្ទុខុសប្លែកពីគេខ្លាំង (outliers) និងទំនាក់ទំនងរវាងជួរឈរនីមួយៗ។ ខ្ញុំបានពិនិត្យមើលការបែងចែកទិន្នន័យ (distributions), ប្រៀបធៀបក្រុមសិស្សតាមកម្រិតវត្តមាន និងពិនិត្យមើលទំនាក់ទំនងឆ្លុះ (correlations)។

---

## **6. Calculate descriptive statistics (គណនាស្ថិតិពិពណ៌នា)**

**🗣️ English Script:**

> For the **398 cleaned records**, the **mean final score is 78.11**, the **median is 79.40**, the **minimum is 42.5**, and the **maximum is 100**. The standard deviation is **11.96**, which tells us how widely the scores vary around the average. I saved the full statistics in [`reports/tables/descriptive_stats.csv`](../reports/tables/descriptive_stats.csv).

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**

> សម្រាប់កំណត់ត្រាទិន្នន័យដែលបានសម្អាតរួចទាំង **៣៩៨ ជួរ**៖ ពិន្ទុបញ្ចប់ជា **មធ្យមភាគ (mean) គឺ ៧៨.១១**, **មេដ្យាន (median) គឺ ៧៩.៤០**, **ទាបបំផុត (minimum) គឺ ៤២.៥**, និង **ខ្ពស់បំផុត (maximum) គឺ ១០០**។ គម្លាតស្តង់ដារ (standard deviation) គឺ **១១.៩៦** ដែលបង្ហាញពីទំហំគម្លាតរាយប៉ាយនៃពិន្ទុធៀបនឹងតម្លៃមធ្យមភាគ។ ខ្ញុំបានរក្សាទុកតារាងស្ថិតិពេញលេញនៅក្នុង [`reports/tables/descriptive_stats.csv`](../reports/tables/descriptive_stats.csv)។

---

## **7. Create at least five visualizations (បង្កើតគំនូសតាងរូបភាពយ៉ាងតិចចំនួន ៥)**

**🗣️ English Script:**

> My project contains **eight Part A figures**, more than the five required. The five main charts are:
>
> 1. The **[final-score histogram](../reports/figures/fig01_final_score_hist.png)** shows where scores are concentrated.
> 2. The **[correlation heatmap](../reports/figures/fig02_corr_heatmap.png)** compares relationships between academic variables.
> 3. The **[midterm-versus-final scatter plot](../reports/figures/fig03_midterm_vs_final.png)** shows that higher midterm scores generally appear with higher final scores.
> 4. The **[attendance box plot](../reports/figures/fig04_attendance_box.png)** compares final scores for low, medium, and high attendance groups.
> 5. The **[study-hours scatter plot](../reports/figures/fig05_study_vs_final.png)** shows a positive relationship between study time and final score.
>
> I also created a [pair plot](../reports/figures/fig06_pairplot.png), an [outlier plot](../reports/figures/fig07_outliers.png), and a [GPA-versus-final plot](../reports/figures/fig08_gpa_vs_final.png).

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**

> គម្រោងរបស់ខ្ញុំមានរូបភាពក្រាហ្វិកសម្រាប់ផ្នែក A ចំនួន **៨ រូបភាព** (លើសពី ៥ រូបភាពដែលតម្រូវ)។ ក្រាហ្វិកចម្បងទាំង ៥ រួមមាន៖
>
> 1. **[Final-score histogram](../reports/figures/fig01_final_score_hist.png)**៖ បង្ហាញពីដង់ស៊ីតេនៃការប្រមូលផ្តុំពិន្ទុរបស់សិស្ស។
> 2. **[Correlation heatmap](../reports/figures/fig02_corr_heatmap.png)**៖ ផែនទីកម្តៅប្រៀបធៀបទំនាក់ទំនងរវាងអថេរសិក្សាទាំងអស់។
> 3. **[Midterm-versus-final scatter plot](../reports/figures/fig03_midterm_vs_final.png)**៖ បង្ហាញថា សិស្សដែលមានពិន្ទុពាក់កណ្តាលឆមាសខ្ពស់ ជាទូទៅទទួលបានពិន្ទុប្រឡងបញ្ចប់ខ្ពស់ដូចគ្នា។
> 4. **[Attendance box plot](../reports/figures/fig04_attendance_box.png)**៖ ប្រៀបធៀបពិន្ទុបញ្ចប់រវាងក្រុមសិស្សដែលមានវត្តមាន ទាប, មធ្យម និងខ្ពស់។
> 5. **[Study-hours scatter plot](../reports/figures/fig05_study_vs_final.png)**៖ បង្ហាញពីទំនាក់ទំនងវិជ្ជមានរវាងរយៈពេលម៉ោងស្វ័យសិក្សា និងពិន្ទុប្រឡងបញ្ចប់។
>
> ក្រៅពីនេះ ខ្ញុំក៏បានបង្កើត [Pair plot](../reports/figures/fig06_pairplot.png), [Outlier plot](../reports/figures/fig07_outliers.png), និង [GPA-versus-final plot](../reports/figures/fig08_gpa_vs_final.png) បន្ថែមផងដែរ។

---

## **8. Identify important patterns and relationships (កំណត់ទម្រង់លំនាំ និងទំនាក់ទំនងសំខាន់ៗ)**

**🗣️ English Script:**

> My strongest observed correlation is between **midterm score and final score: 0.894**. Assignment average is **0.854**, previous GPA **0.844**, study hours **0.821**, and attendance **0.807**. Students in the high-attendance group also have a higher median final score than students in the low-attendance group. But **correlation does not prove causation**, and these patterns come from synthetic data.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**

> ទំនាក់ទំនងឆ្លុះ (correlation) ខ្លាំងជាងគេបំផុតដែលបានរកឃើញ គឺរវាង **ពិន្ទុពាក់កណ្តាលឆមាស និងពិន្ទុបញ្ចប់៖ ០.៨៩៤ (0.894)**។ មធ្យមភាគកិច្ចការផ្ទះគឺ **០.៨៥៤**, GPA ពីមុន **០.៨៤៤**, ម៉ោងសិក្សា **០.៨២១**, និងវត្តមាន **០.៨០៧**។ សិស្សក្នុងក្រុមដែលមានវត្តមានខ្ពស់ ក៏មានពិន្ទុមេដ្យានបញ្ចប់ខ្ពស់ជាងសិស្សក្នុងក្រុមវត្តមានទាបដែរ។ ប៉ុន្តែ **ទំនាក់ទំនងស្ថិតិ (Correlation) មិនមែនជាការបញ្ជាក់ពីហេតុនិងផល (Causation) ទាំងស្រុងនោះទេ** ហើយទិន្នន័យទាំងនេះបានមកពីសំណុំទិន្នន័យសិប្បនិម្មិត។

---

## **Conclusion (សេចក្តីសន្និដ្ឋាន)**

**🗣️ English Script:**

> In conclusion, I cleaned **402 records down to 398**, handled missing values, calculated descriptive statistics, prepared **eight visualizations**, and explained the main relationships. This completes the teacher's **eight Part A requirements**. Thank you.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**

> ជារួមមក ខ្ញុំបានសម្អាតទិន្នន័យពី **៤០២ ជួរ មកនៅត្រឹម ៣៩៨ ជួរ**, ដោះស្រាយបញ្ហាទិន្នន័យដែលបាត់, គណនាស្ថិតិពិពណ៌នា, បង្កើត **រូបភាពក្រាហ្វិកវិភាគចំនួន ៨**, និងបានពន្យល់ពីទំនាក់ទំនងសំខាន់ៗទាំងអស់។ នេះបានបំពេញគ្រប់ **លក្ខខណ្ឌតម្រូវទាំង ៨ នៃផ្នែក A** របស់លោកគ្រូ/អ្នកគ្រូ។ សូមអរគុណ!

---

## **Possible teacher questions and short answers (សំណួរដែលអាចសួរដោយគ្រូ និងចម្លើយខ្លីៗ)**

**Q1. Are these real student records?**\
_(តើនេះជាទិន្នន័យសិស្សពិតប្រាកដមែនទេ?)_\
**Ans (EN):** No. They are synthetic benchmark records generated by [`src/data_loader.py`](../src/data_loader.py), so the conclusions are demonstrations of data-analysis methods.\
**ចម្លើយ (KH):** មិនមែនទេលោកគ្រូ/អ្នកគ្រូ។ នេះជាទិន្នន័យសិប្បនិម្មិត (Synthetic benchmark) ដែលបង្កើតឡើងដោយកូដ [`src/data_loader.py`](../src/data_loader.py) សម្រាប់បង្ហាញអំពីវិធីសាស្ត្រក្នុងការវិភាគទិន្នន័យតែប៉ុណ្ណោះ។

**Q2. Why did 402 rows become 398?**\
_(ហេតុអ្វីបានជា ៤០២ ជួរ ក្លាយជា ៣៩៨ ជួរ?)_\
**Ans (EN):** Two duplicate-ID rows and two rows missing the final score were removed.\
**ចម្លើយ (KH):** ពីព្រោះយើងបានលុបទិន្នន័យដែលស្ទួន ID ចំនួន ២ ជួរ និងទិន្នន័យដែលអត់មានពិន្ទុបញ្ចប់ (final score) ចំនួន ២ ជួរចេញ។

**Q3. Why use both mean and median?**\
_(ហេតុអ្វីបានជាប្រើទាំង Mean និង Median?)_\
**Ans (EN):** Mean is the arithmetic average; median is the middle value. Median is less influenced by extreme values and was used for the more skewed assignment scores.\
**ចម្លើយ (KH):** Mean គឺជាមធ្យមភាគនព្វន្ធ ចំណែក Median គឺជាតម្លៃចំកណ្តាល។ Median មិនសូវរងឥទ្ធិពលពីតម្លៃដែលខុសគេខ្លាំងពេកទេ (Extreme values/Outliers) ដូច្នេះហើយទើបយកវាមកប្រើលើពិន្ទុកិច្ចការផ្ទះដែលមានភាពលម្អៀង។

**Q4. Why remove rows with missing final scores?**\
_(ហេតុអ្វីត្រូវលុបជួរដែលបាត់ពិន្ទុបញ្ចប់ (final score) ចោល?)_\
**Ans (EN):** Because final score is the main value being analyzed. Guessing it would make the analysis less trustworthy.\
**ចម្លើយ (KH):** ពីព្រោះពិន្ទុបញ្ចប់ (final score) គឺជាអថេរគោលដៅចម្បងដែលយើងត្រូវយកមកវិភាគ។ ការស្មានបំពេញពិន្ទុនេះអាចធ្វើឱ្យការវិភាគបាត់បង់ភាពសុក្រឹត និងមិនគួរឱ្យទុកចិត្ត។

**Q5. Which variable has the strongest correlation?**\
_(តើអថេរមួយណាមានទំនាក់ទំនង (Correlation) ខ្លាំងជាងគេ?)_\
**Ans (EN):** Midterm score, with Pearson correlation **0.894** in the cleaned dataset.\
**ចម្លើយ (KH):** គឺពិន្ទុពាក់កណ្តាលឆមាស (Midterm score) ដែលមាន Pearson correlation ខ្ពស់រហូតដល់ **០.៨៩៤ (0.894)**។

**Q6. Does correlation mean that studying more causes better scores?**\
_(តើ Correlation មានន័យថា រៀនកាន់តែច្រើន ធ្វើឱ្យពិន្ទុកាន់តែល្អដោយស្វ័យប្រវត្តមែនទេ?)_\
**Ans (EN):** No. Correlation describes an association, not proof of cause and effect.\
**ចម្លើយ (KH):** មិនមែនទេបាទ/ចាស។ Correlation គ្រាន់តែពណ៌នាអំពីទំនាក់ទំនងរវាងអថេរពីរប៉ុណ្ណោះ វាមិនមែនជាភស្តុតាងបញ្ជាក់ពីហេតុនិងផល (Cause and effect) នោះទេ។

**Q7. What does standard deviation mean?**\
_(តើគម្លាតស្តង់ដារ (Standard Deviation) បញ្ជាក់ពីអ្វី?)_\
**Ans (EN):** It measures spread around the mean. For final scores, it is **11.96 points**.\
**ចម្លើយ (KH):** វាវាស់វែងពីកម្រិតរាយប៉ាយនៃទិន្នន័យធៀបនឹងតម្លៃមធ្យមភាគ។ សម្រាប់ពិន្ទុបញ្ចប់ គម្លាតស្តង់ដារគឺ **១១.៩៦ ពិន្ទុ**។

**Q8. Where are your figures?**\
_(តើរូបភាពក្រាហ្វិករបស់អ្នកនៅឯណា?)_\
**Ans (EN):** The eight Part A image files are in [`reports/figures/`](../reports/figures/).\
**ចម្លើយ (KH):** រូបភាពក្រាហ្វិកទាំង ៨ នៃផ្នែក A ត្រូវបានរក្សាទុកនៅក្នុងថត [`reports/figures/`](../reports/figures/)។

**Q9. Why do older reports have different correlation numbers?**\
_(ហេតុអ្វីបានជារបាយការណ៍ចាស់ៗមានលេខ Correlation ខុសពីនេះ?)_\
**Ans (EN):** Some older narrative summaries are inconsistent. I checked the cleaned CSV and use the verified values in the [Part A guide](PART_A_DATA_ANALYSIS_GUIDE.md).\
**ចម្លើយ (KH):** អត្ថបទសង្ខេបចាស់ៗខ្លះមិនទាន់ស្របគ្នាទាំងស្រុង។ ខ្ញុំបានពិនិត្យមើលផ្ទៀងផ្ទាត់ឡើងវិញពីទិន្នន័យ CSV ដែលបានសម្អាតរួច និងប្រើប្រាស់ទិន្នន័យត្រឹមត្រូវដែលបានផ្ទៀងផ្ទាត់នៅក្នុង [Part A Guide](PART_A_DATA_ANALYSIS_GUIDE.md)។

**Q10. How do you reproduce Part A?**\
_(តើអ្នកដំណើរការកូដផ្នែក A ឡើងវិញដោយរបៀបណា?)_\
**Ans (EN):** From the repository root, install the requirements, run `python src/cleaning.py`, then run `python src/eda.py`. Those are the existing project scripts.\
**ចម្លើយ (KH):** ចេញពី Root folder នៃ Project យើងគ្រាន់តែដំឡើងបណ្ណាល័យដែលត្រូវការ រួចដំណើរការ `python src/cleaning.py` និងបន្តដំណើរការ `python src/eda.py`។ ទាំងនេះជាស្គ្រីបដែលមានស្រាប់ក្នុងគម្រោង។
