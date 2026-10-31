# **Part B – Student Presentation Script (អត្ថបទបទបង្ហាញ ផ្នែក B - ទ្វេភាសា អង់គ្លេស-ខ្មែរ)**

**Topic / ប្រធានបទ:** Student Record Prediction — Part B Machine Learning (ការទស្សន៍ទាយពិន្ទុសិស្ស — ផ្នែក B៖ ការរៀនរបស់ម៉ាស៊ីន)  
**Speaking time / រយៈពេលនិយាយ:** About 5–7 minutes at a clear, comfortable pace / ប្រហែល ៥-៧ នាទី  
**Reference / ឯកសារយោង:** [Part B Machine Learning Guide](PART_B_MACHINE_LEARNING_GUIDE.md) | [Part A Guide](PART_A_DATA_ANALYSIS_GUIDE.md)

---

## **1. Define the Target Variable (កំណត់អថេរគោលដៅ)**

**🗣️ English Script:**
> Good morning, teacher and classmates. Today, I am presenting **Part B: Machine Learning** of my Student Record Prediction project. Building upon the cleaned dataset from Part A, our machine learning task is **Supervised Regression**. 
>
> Our target variable is **`final_score`**, a continuous numerical grade ranging from $0$ to $100$ points. We deliberately chose regression over binary classification (pass/fail) because educators need granular point predictions to detect slight declines before students fail. To prevent data leakage, `final_score` is strictly isolated from the input features.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**
> ជម្រាបសួរលោកគ្រូ អ្នកគ្រូ និងមិត្តរួមថ្នាក់ទាំងអស់គ្នា។ ថ្ងៃនេះ ខ្ញុំសូមធ្វើបទបង្ហាញអំពី **ផ្នែក B៖ ការរៀនរបស់ម៉ាស៊ីន (Part B: Machine Learning)** នៃគម្រោងទស្សន៍ទាយពិន្ទុសិស្ស។ ដោយបន្តពីសំណុំទិន្នន័យដែលបានសម្អាតរួចក្នុងផ្នែក A ការងារ Machine Learning របស់យើងគឺ **Supervised Regression (តម្រែតម្រង់ក្រោមការត្រួតពិនិត្យ)**។
>
> អថេរគោលដៅ (Target Variable) របស់យើងគឺ **`final_score`** ដែលជាពិន្ទុជាលេខបន្ត (continuous) ចន្លោះពី ០ ដល់ ១០០ ពិន្ទុ។ យើងជ្រើសរើសយក Regression ជាជាងការបែងចែកជាធ្លាក់ឬជាប់ (Pass/Fail) ពីព្រោះលោកគ្រូអ្នកគ្រូត្រូវការដឹងពីពិន្ទុលម្អិតជាក់ស្តែង ដើម្បីអន្តរាគមន៍បានទាន់ពេលវេលា។ ដើម្បីការពារកុំឱ្យបែកធ្លាយទិន្នន័យ (Data Leakage) អថេរ `final_score` ត្រូវបានដកចេញដាច់ដោយឡែកពី Input Features។

---

## **2. Select Relevant Features (ជ្រើសរើសលក្ខណៈពិសេសដែលពាក់ព័ន្ធ)**

**🗣️ English Script:**
> We selected **5 input features** representing key dimensions of student academic behavior:
> 1. `attendance_pct` (class attendance rate)
> 2. `study_hours_week` (independent weekly self-study)
> 3. `assignment_avg` (homework and quiz average)
> 4. `midterm_score` (midterm examination performance)
> 5. `previous_gpa` (cumulative GPA from prior terms)
>
> To check for multicollinearity, we calculated the **Variance Inflation Factor (VIF)**. All VIF scores are below **$4.3$**, well beneath the risk threshold of $5.0$. This proves each feature brings unique, independent information, so all five features are retained.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**
> យើងបានជ្រើសរើស **Features ចំនួន ៥** ដែលឆ្លុះបញ្ចាំងពីទិដ្ឋភាពសំខាន់ៗនៃការសិក្សារបស់សិស្ស៖
> 1. `attendance_pct` (អត្រាវត្តមានរៀនក្នុងថ្នាក់)
> 2. `study_hours_week` (ម៉ោងស្វ័យសិក្សាប្រចាំសប្តាហ៍)
> 3. `assignment_avg` (ពិន្ទុមធ្យមភាគកិច្ចការផ្ទះ)
> 4. `midterm_score` (ពិន្ទុប្រឡងពាក់កណ្តាលឆមាស)
> 5. `previous_gpa` (និទ្ទេស GPA សរុបពីឆមាសមុនៗ)
>
> ដើម្បីបញ្ជាក់ថា Features ទាំងនេះមិនជាន់គ្នាខ្លាំង (multicollinearity) យើងបានគណនា **VIF (Variance Inflation Factor)**។ តម្លៃ VIF ទាំងអស់គឺទាបជាង **៤.៣** ដែលទាបជាងកម្រិតប្រកាសអាសន្ន ៥.០។ នេះបញ្ជាក់ថា Feature នីមួយៗផ្តល់ព័ត៌មានឯករាជ្យរៀងៗខ្លួន ដូច្នេះយើងរក្សាទុក Features ទាំង ៥ នេះទាំងអស់។

---

## **3. Preprocess the Data (រៀបចំនិងកែច្នៃទិន្នន័យជាមុន)**

**🗣️ English Script:**
> Because our features have different scales—like GPA from $0$ to $4.0$, study hours up to $50$, and test scores up to $100$—we standardized them using **`StandardScaler`** to give them zero mean and unit variance.
>
> Crucially, to prevent data leakage, we encapsulated the scaler inside a **scikit-learn Pipeline**. The scaler is fitted strictly on the training set only, and then applied to transform the test set.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**
> ដោយសារតែ Features របស់យើងមានខ្នាតខុសៗគ្នា ដូចជា GPA គិតពី ០ ដល់ ៤.០, ម៉ោងសិក្សាគិតដល់ ៥០ ម៉ោង និងពិន្ទុប្រឡងគិតដល់ ១០០ យើងបានធ្វើស្តង់ដារកម្មទិន្នន័យដោយប្រើ **`StandardScaler`** (z-score normalization)។
>
> អ្វីដែលសំខាន់បំផុត ដើម្បីកុំឱ្យទិន្នន័យតេស្តជ្រាបចូលទៅក្នុងម៉ូដែល យើងបានខ្ចប់ Scaler នេះនៅក្នុង **Pipeline របស់ scikit-learn**។ Scaler នេះត្រូវបានគណនាតែលើ Training Set ប៉ុណ្ណោះ ហើយទើបយកទៅអនុវត្តលើ Test Set ជាក្រោយ។

---

## **4. Split the Dataset: 80% Train / 20% Test (បែងចែកសំណុំទិន្នន័យ)**

**🗣️ English Script:**
> We split our $398$ clean student records using an **80/20 ratio**:
> * **Training Set (80%):** $318$ students
> * **Testing Set (20%):** $80$ students
>
> We set a fixed seed of `random_state = 42` for exact reproducibility. Both splits have nearly identical average scores—$78.1$ points for training and $77.9$ points for testing—verifying a balanced, unbiased partition.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**
> យើងបានបែងចែកទិន្នន័យសិស្សដែលស្អាតទាំង ៣៩៨ នាក់ តាមសមាមាត្រ **៨០% និង ២០%**៖
> * **Training Set (៨០%):** សិស្សចំនួន ៣១៨ នាក់
> * **Testing Set (២០%):** សិស្សចំនួន ៨០ នាក់
>
> យើងបានកំណត់ Random Seed ថេរស្មើនឹង ៤២ ដើម្បីឱ្យលទ្ធផលអាចផ្ទៀងផ្ទាត់ឡើងវិញបានយ៉ាងជាក់លាក់។ សំណុំទិន្នន័យទាំងពីរមានមធ្យមភាគពិន្ទុប្រហាក់ប្រហែលគ្នាខ្លាំង (៧៨.១ ពិន្ទុក្នុង Train និង ៧៧.៩ ពិន្ទុក្នុង Test) ដែលបញ្ជាក់ពីការបែងចែកប្រកបដោយតុល្យភាព។

---

## **5. Train At Least 2 Machine Learning Models (បង្វឹកម៉ូដែលយ៉ាងតិច ២)**

**🗣️ English Script:**
> We trained two different machine learning models to contrast parametric simplicity against non-linear ensemble flexibility:
> 1. **Multiple Linear Regression (OLS):** Our parametric baseline that finds direct linear coefficients.
> 2. **Random Forest Regressor:** A non-linear ensemble model with $100$ bagged decision trees and a maximum depth of $6$.
> 
> We also trained a **Ridge Regression** model as a regularized baseline to test for coefficient shrinkage.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**
> យើងបានបង្វឹកម៉ូដែល Machine Learning ចំនួន ២ ផ្សេងគ្នា ដើម្បីប្រៀបធៀបរវាងម៉ូដែលសាមញ្ញ (parametric) និងម៉ូដែលស្មុគស្មាញ (non-linear ensemble)៖
> 1. **Multiple Linear Regression (OLS):** ជាម៉ូដែលគោលបែបលីនេអ៊ែរ ដែលផ្តល់មេគុណច្បាស់លាស់ និងងាយស្រួលបកស្រាយ។
> 2. **Random Forest Regressor:** ជាម៉ូដែល Ensemble ដែលបូកបញ្ចូលដើមឈើសេចក្តីសម្រេច (Decision Trees) ចំនួន ១០០ ដើម ដោយកំណត់ជម្រៅអតិបរមា Depth ស្មើ ៦។
>
> បន្ថែមពីនេះ យើងក៏បានបង្វឹក **Ridge Regression** មួយទៀតជាម៉ូដែល Regularization ផងដែរ។

---

## **6. Compare the Models (ប្រៀបធៀបម៉ូដែល)**

**🗣️ English Script:**
> Looking at our evaluation results on the unseen test set:
> * **Multiple Linear Regression:** Achieved a **Test RMSE of 3.39**, **Test MAE of 2.63**, and **Test $R^2$ of 0.929**.
> * **Random Forest Regressor:** Achieved a **Test RMSE of 4.02**, **Test MAE of 3.15**, and **Test $R^2$ of 0.900**.
> * In **5-Fold Cross-Validation**, Linear Regression also achieved a higher average $R^2$ of **0.881** versus **0.854** for Random Forest.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**
> ផ្អែកលើលទ្ធផលនៃការវាយតម្លៃលើសំណុំទិន្នន័យតេស្តដែលម៉ូដែលមិនធ្លាប់ឃើញពីមុន (Holdout Test Set)៖
> * **Multiple Linear Regression:** ទទួលបាន **Test RMSE ស្មើ ៣.៣៩**, **Test MAE ស្មើ ២.៦៣**, និង **Test $R^2$ ស្មើ ០.៩២៩**។
> * **Random Forest Regressor:** ទទួលបាន **Test RMSE ស្មើ ៤.០២**, **Test MAE ស្មើ ៣.១៥**, និង **Test $R^2$ ស្មើ ០.៩០០**។
> * ក្នុងការផ្ទៀងផ្ទាត់ឆ្លងកាត់ ៥-Fold Cross-Validation, Linear Regression ក៏ទទួលបានមធ្យមភាគ $R^2$ ខ្ពស់ជាងដែរ គឺ **០.៨៨១** ធៀបនឹង **០.៨៥៤** របស់ Random Forest។

---

## **7. Evaluate Model Performance (វាយតម្លៃសមត្ថភាពម៉ូដែល)**

**🗣️ English Script:**
> The evaluation metrics reveal an important machine learning lesson:
> * An MAE of **$2.63$ points** for Linear Regression means predictions deviate by less than three points on a 100-point scale.
> * Random Forest showed clear evidence of **overfitting**: on training data, its $R^2$ was nearly perfect at **$0.981$**, but dropped to **$0.900$** on test data.
> * Linear Regression showed no overfitting, generalizing reliably across all cross-validation folds.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**
> រង្វាស់នៃការវាយតម្លៃបានបង្ហាញនូវចំណុចសំខាន់ក្នុង Machine Learning៖
> * តម្លៃ MAE ត្រឹមតែ **២.៦៣ ពិន្ទុ** បញ្ជាក់ថាការទស្សន៍ទាយរបស់ Linear Regression ខុសគ្នាជាមធ្យមតិចជាង ៣ ពិន្ទុប៉ុណ្ណោះលើពិន្ទុពេញ ១០០។
> * Random Forest បានបង្ហាញសញ្ញានៃការ **Overfitting (ទន្ទេញចាំទិន្នន័យចាស់)**៖ នៅលើ Train Data វាធ្វើបាន $R^2$ ដល់ទៅ **០.៩៨១** ប៉ុន្តែធ្លាក់ចុះមកសល់ត្រឹម **០.៩០០** នៅលើ Test Data។
> * ចំណែកឯ Linear Regression មិនមានបញ្ហា Overfitting ទេ ហើយមានស្ថិរភាពខ្ពស់នៅគ្រប់ Fold ទាំងអស់។

---

## **8. Select the Model Based on Evaluation Metrics (ជ្រើសរើសម៉ូដែលផ្អែកលើរង្វាស់)**

**🗣️ English Script:**
> We selected **Multiple Linear Regression** as our production model for four clear reasons:
> 1. **Lower Error:** It had a lower test RMSE ($3.39$ vs $4.02$).
> 2. **Cross-Validation Stability:** It achieved higher stability and lower variance across folds.
> 3. **Occam’s Razor:** When a simpler model outperforms a complex one, data science principles dictate choosing the simpler model.
> 4. **Explainability:** Teachers can directly understand and use linear weights.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**
> យើងបានសម្រេចជ្រើសរើសយក **Multiple Linear Regression** ជាម៉ូដែលផ្លូវការ ដោយសារហេតុផលចម្បង ៤ យ៉ាង៖
> 1. **កំហុសទាបជាង (Lower Error):** វាមាន Test RMSE ទាបជាង Random Forest (៣.៣៩ ធៀបនឹង ៤.០២)។
> 2. **ស្ថិរភាព Cross-Validation:** វាមានលទ្ធផលខ្ពស់ និងរឹងមាំជាងនៅគ្រប់ Fold។
> 3. **គោលការណ៍ Occam’s Razor:** ក្នុងវិទ្យាសាស្ត្រទិន្នន័យ ប្រសិនបើម៉ូដែលសាមញ្ញផ្តល់លទ្ធផលល្អជាងម៉ូដែលស្មុគស្មាញ យើងត្រូវតែជ្រើសរើសម៉ូដែលសាមញ្ញ។
> 4. **ភាពងាយស្រួលបកស្រាយ (Interpretability):** លោកគ្រូអ្នកគ្រូអាចមើលយល់ពីទម្ងន់មេគុណនីមួយៗបានយ៉ាងច្បាស់លាស់។

---

## **9. Make Predictions (ធ្វើការទស្សន៍ទាយ)**

**🗣️ English Script:**
> We deployed our model to generate predictions on the $80$ holdout test students:
> * For example, student `STU_0201` had an actual score of **$90.1$**, and our model predicted **$87.6$**, an error of just $2.5$ points.
> * For student `STU_0211`, the actual score was **$89.2$** and the prediction was **$89.0$**, almost an exact match.
> * **$90\%$ of all test students** were predicted within $\pm 5.4$ points of their actual grades. Our residual diagnostic plots confirm that errors are normally distributed around zero.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**
> យើងបានប្រើម៉ូដែលនេះដើម្បីទស្សន៍ទាយពិន្ទុសិស្សតេស្តទាំង ៨០ នាក់៖
> * ឧទាហរណ៍ សិស្សកូដ `STU_0201` មានពិន្ទុជាក់ស្តែង **៩០.១** ហើយម៉ូដែលទស្សន៍ទាយបាន **៨៧.៦** (ខុសគ្នាត្រឹម ២.៥ ពិន្ទុ)។
> * សិស្សកូដ `STU_0211` មានពិន្ទុជាក់ស្តែង **៨៩.២** ហើយម៉ូដែលទស្សន៍ទាយបាន **៨៩.០** ដែលស្ទើរតែដូចគ្នាបេះបិទ។
> * **៩០% នៃសិស្សទាំងអស់** ត្រូវបានទស្សន៍ទាយក្នុងគម្លាតត្រឹម $\pm ៥.៤$ ពិន្ទុប៉ុណ្ណោះ។ ក្រាហ្វ Residuals បញ្ជាក់ថាកំហុសទាំងអស់មានលក្ខណៈធម្មតា (Normal Distribution) ជុំវិញសូន្យ។

---

## **10. Explain the Results (ពន្យល់បកស្រាយលទ្ធផល)**

**🗣️ English Script:**
> Our model's standardized coefficients show which factors matter most:
> 1. **Midterm Score (+4.06 std):** The strongest predictor. Performance at mid-semester is the best indicator of the final exam.
> 2. **Study Hours (+2.19 std):** Second strongest. In practical terms, each additional weekly study hour contributes about **$+0.40$ points** to the final score.
> 3. **Previous GPA (+2.06 std), Assignment Average (+2.03 std), and Attendance (+1.98 std)** all provide steady positive gains.
>
> **Important Academic Note:** These coefficients show statistical correlation within our dataset, but **correlation does not prove causation**. Studying more hours is associated with higher marks, but cannot guarantee them.
>
> Thank you, teacher. I am now ready for your questions.

**🇰🇭 អត្ថន័យជាភាសាខ្មែរ (Khmer Translation):**
> មេគុណស្តង់ដារ (Standardized Coefficients) នៃម៉ូដែលបង្ហាញពីកម្រិតឥទ្ធិពលនៃកត្តានីមួយៗ៖
> 1. **ពិន្ទុពាក់កណ្តាលឆមាស (+៤.០៦):** ជាកត្តាជះឥទ្ធិពលខ្លាំងជាងគេបំផុត។
> 2. **ម៉ោងស្វ័យសិក្សា (+២.១៩):** ជះឥទ្ធិពលខ្លាំងលំដាប់ទីពីរ។ ជាការអនុវត្តជាក់ស្តែង រាល់ការបង្កើនម៉ោងសិក្សា ១ ម៉ោងក្នុងមួយសប្តាហ៍ ជួយបង្កើនពិន្ទុប្រហែល **+០.៤០ ពិន្ទុ**។
> 3. **GPA ពីមុន (+២.០៦), កិច្ចការផ្ទះ (+២.០៣), និងវត្តមាន (+១.៩៨)** សុទ្ធតែជួយរុញច្រានពិន្ទុឱ្យកើនឡើងជាវិជ្ជមាន។
>
> **កំណត់សម្គាល់បែបវិទ្យាសាស្ត្រ៖** លទ្ធផលទាំងនេះជាទំនាក់ទំនងស្ថិតិ (Correlation) មិនមែនជាការសន្និដ្ឋានហេតុនិងផលដាច់ខាត (Causation) ឡើយ។
>
> សូមអរគុណលោកគ្រូ អ្នកគ្រូ។ ខ្ញុំរួចរាល់សម្រាប់ការឆ្លើយសំណួរហើយបាទ/ចាស។

---

## **Teacher Q&A Defense (សំណួរពិបាកៗដែលគ្រូអាចសួរ និងរបៀបឆ្លើយការពារ)**

| Possible Teacher Question | Recommended Defense & Spoken Answer |
| :--- | :--- |
| **Q1: Why didn't you classify final scores into Pass / Fail?** | *"Classifying into binary Pass/Fail throws away valuable numerical variance. A student dropping from 85 to 65 is in serious trouble, but binary classification would treat both as 'Pass'. Regression provides exact point predictions for early intervention."* |
| **Q2: Why did Linear Regression beat Random Forest? Isn't Random Forest more advanced?** | *"Random Forest is flexible, but when the underlying data-generating relationship is predominantly linear additive, decision trees suffer from high variance and step-function approximations. On 318 training rows, Random Forest overfitted ($R^2=0.98$ train vs $0.90$ test), whereas Linear Regression generalized better."* |
| **Q3: How did you ensure there was no data leakage during feature scaling?** | *"We wrapped `StandardScaler` inside a scikit-learn `Pipeline`. The scaler was strictly fitted on the 80% training split only (`fit_transform`) and then applied to transform the test split without refitting."* |
| **Q4: How did you verify your linear regression assumptions?** | *"We inspected residual diagnostic plots (`fig09_residuals.png` and `fig10_actual_vs_pred.png`). Residuals are centered at zero ($\mu_e = -0.198$), show homoscedasticity across predicted score tiers, and follow a Gaussian normal distribution."* |
| **Q5: Can you claim that increasing study hours causes students to pass?** | *"No, we must maintain scientific integrity. Regression models demonstrate predictive association, not counterfactual causality. Unobserved factors like student motivation or prior knowledge may confound the relationship."* |
