

# Data record model

# Line one of dataset
# Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease
class Record:
    def __init__(self):
        self.age = 0
        self.sex = 0
        self.chest_pain_type = []
        self.resting_bp = 0
        self.cholesterol = 0
        self.fasting_bs = 0
        self.resting_ecg = []
        self.max_hr = 0
        self.exercise_angina = 0
        self.oldpeak = 0
        self.st_slope = []
        self.heart_disease = 0

    def get_target(self):
        return self.heart_disease

    def get_feature_vector(self):

        feature_vector = [self.age,
                          self.sex,
                          self.chest_pain_type[0],
                          self.chest_pain_type[1],
                          self.resting_bp,
                          self.cholesterol,
                          self.fasting_bs,
                          self.resting_ecg[0],
                          self.resting_ecg[1],
                          self.max_hr,
                          self.exercise_angina,
                          self.oldpeak,
                          self.st_slope[0],
                          self.st_slope[1]

                          ]
        return feature_vector

    def process_record(self, rec):
        self.age = self.__age_vector(rec[0])
        self.sex = self.__sex_vector(rec[1])
        self.chest_pain_type = self.__chest_pain_type_vector(rec[2])
        self.resting_bp = self.__resting_bp_vector(rec[3])
        self.cholesterol = self.__cholesterol_vector(rec[4])
        self.fasting_bs = self.__fasting_bs_vector(rec[5])
        self.resting_ecg = self.__resting_ecg_vector(rec[6])
        self.max_hr = self.__max_hr_vector(rec[7])
        self.exercise_angina = self.__exercise_angina_vector(rec[8])
        self.oldpeak = self.__oldpeak_vector(rec[9])
        self.st_slope = self.__st_slope_vector(rec[10])
        self.heart_disease = self.__heart_disease_vector(rec[11])

    # 1-d vector is a scalar. This step is not really necessary, but is correct in it's placement.
    def __age_vector(self, age):
        age = int(age)
        age_range = [20, 80]
        age = (age - age_range[0]) / (age_range[1] - age_range[0])
        return abs(age)

    def __sex_vector(self, sex):
        if sex == ['M']:
            return 1
        else:
            return 0

    def __chest_pain_type_vector(self, chest_pain_type):
        if chest_pain_type == 'TA':
            return 0, 0
        elif chest_pain_type == 'ATA':
            return 0, 1
        elif chest_pain_type == 'NAP':
            return 1, 0
        elif chest_pain_type == 'ASY':
            return 1, 1
        else:
            raise Exception("[__chest_pain_type_vector] : Error parsing vector")

    def __resting_bp_vector(self, resting_bp):
        resting_bp = int(resting_bp)
        bp_range = [200, 0]
        resting_bp = (resting_bp - bp_range[0]) / (bp_range[1] - bp_range[0])
        return abs(resting_bp)

    def __cholesterol_vector(self, cholesterol):
        cholesterol = int(cholesterol)
        cholesterol_range = [603, 0]
        if cholesterol > cholesterol_range[0] or cholesterol < cholesterol_range[1]:
            raise Exception("[__cholesterol_vector] : Heart rate out of range")
        cholesterol = (cholesterol - cholesterol_range[0]) / (cholesterol_range[1] - cholesterol_range[0])
        return abs(cholesterol)

    def __fasting_bs_vector(self, fasting_bs):
        return int(fasting_bs)

    def __resting_ecg_vector(self, resting_ecg):
        if resting_ecg == 'Normal':
            return 0, 0
        elif resting_ecg == 'ST':
            return 0, 1
        elif resting_ecg == 'ST-T':
            return 1, 0
        elif resting_ecg == 'LVH':
            return 1, 1
        else:
            raise Exception("[__resting_ecg_vector] : Error parsing vector")

    def __max_hr_vector(self, max_hr):
        max_hr = int(max_hr)
        max_hr_range = [60, 202]
        if max_hr > max_hr_range[1] or max_hr < max_hr_range[0]:
            raise Exception("[__max_hr_vector] : Heart rate out of range")
        max_hr = (max_hr - max_hr_range[0]) / (max_hr_range[1] - max_hr_range[0])
        return abs(max_hr)

    def __exercise_angina_vector(self, exercise_angina):
        if exercise_angina == 'Y':
            return 1
        elif exercise_angina == 'N':
            return 0
        else:
            raise Exception("[__exercise_angina_vector] : Error parsing vector")

    def __oldpeak_vector(self, oldpeak):
        oldpeak = float(oldpeak)
        oldpeak_range = [-2.6, 6.2]
        oldpeak = (oldpeak - oldpeak_range[0]) / (oldpeak_range[1] - oldpeak_range[0])
        return oldpeak

    def __st_slope_vector(self, st_vector):
        if st_vector == 'Up':
            return 0, 0
        elif st_vector == 'Flat':
            return 0, 1
        elif st_vector == 'Down':
            return 1, 0
        else:
            raise Exception("[__st_slope_vector] : Error parsing vector")

    def __heart_disease_vector(self, heart_disease):
        return int(heart_disease)
