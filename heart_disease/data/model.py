

# Data record model

# Line one of dataset
# Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease
class Record:
    def __init__(self):
        self.age = 0
        self.sex = 0
        self.chest_pain_type = 0
        self.resting_bp = 0
        self.cholesterol = 0
        self.fasting_bs = 0
        self.max_hr = 0
        self.exercise_angina = 0
        self.oldpeak = 0
        self.st_slope = 0
        self.heart_disease = 0

    def process_record(self, rec):
        self.age = self.__age_vector(rec[0])
        self.sex = self.__sex_vector(rec[1])
        self.chest_pain_type = self.__chest_pain_type_vector(rec[2])
        self.resting_bp = self.__resting_bp_vector(rec[3])
        self.cholesterol = self.__cholesterol_vector(rec[4])
        self.fasting_bs = self.__fasting_bs_vector(rec[5])
        self.__resting_ecg_vector(rec[6])
        self.max_hr = self.__max_hr_vector(rec[7])
        self.exercise_angina = self.__exercise_angina_vector(rec[8])
        self.oldpeak = self.__oldpeak_vector(rec[9])
        self.st_slope = self.__st_slope_vector(rec[10])
        self.heart_disease = self.__heart_disease_vector(rec[11])

    # 1-d vector is a scalar. This step is not really necessary, but is correct in it's placement.
    def __age_vector(self, age):
        return [int(age)]

    def __sex_vector(self, sex):
        if sex == ['M']:
            return [1]
        else:
            return [0]

    def __chest_pain_type_vector(self, chest_pain_type):
        if chest_pain_type == 'TA':
            return [[0], [0]]
        elif chest_pain_type == 'ATA':
            return [[0], [1]]
        elif chest_pain_type == 'NAP':
            return [[1], [0]]
        elif chest_pain_type == 'ASY':
            return [[1], [1]]
        else:
            raise Exception("[chest_pain_type_vector] : Error parsing vector")

    def __resting_bp_vector(self, resting_bp):
        return [int(resting_bp)]

    def __cholesterol_vector(self, cholesterol):
        return [int(cholesterol)]

    def __fasting_bs_vector(self, fasting_bs):
        return [int(fasting_bs)]

    def __resting_ecg_vector(self, resting_ecg):
        if resting_ecg == 'Normal':
            return [[0], [0]]
        elif resting_ecg == 'ST':
            return [[0], [1]]
        elif resting_ecg == 'ST-T':
            return [[1], [0]]
        elif resting_ecg == 'LVH':
            return [[1], [1]]
        else:
            raise Exception("[resting_ecg_vector] : Error parsing vector")

    def __max_hr_vector(self, max_hr):
        max_hr = int(max_hr)
        if max_hr > 202 or max_hr < 60:
            raise Exception("[max_hr_vector] : Heart rate out of range")
        return [max_hr]

    def __exercise_angina_vector(self, exercise_angina):
        if exercise_angina == 'Y':
            return [1]
        elif exercise_angina == 'N':
            return [0]
        else:
            raise Exception("[exercise_angina_vector] : Error parsing vector")

    def __oldpeak_vector(self, oldpeak):
        return [float(oldpeak)]

    def __st_slope_vector(self, st_vector):
        if st_vector == 'Up':
            return [[0], [0]]
        elif st_vector == 'Flat':
            return [[0], [1]]
        elif st_vector == 'Down':
            return [[1], [0]]
        else:
            raise Exception("[st_slope_vector] : Error parsing vector")

    def __heart_disease_vector(self, heart_disease):
        return [int(heart_disease)]
