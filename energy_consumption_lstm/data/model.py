import datetime as dt
import math
import numpy as np


def calculate_ranges(dataset):
    arr = np.array(dataset)
    mean = np.mean(arr, axis=0)
    min = np.min(arr, axis=0)
    max = np.max(arr, axis=0)

    ranges = np.array((min, mean, max)).T
    return ranges


class Record:
    def __init__(self):
        self.time_date = 0
        self.global_active_Ah_min = 0
        self.global_reactive_Ah_min = 0
        self.voltage = 0
        self.current = 0
        self.sub_meters = []
        self.residual_active_energy = 0
        self.error_active = 0
        self.power = 0


    def process_entry(self, arr):
        ret = [self.__date_time_timestamp(arr[0], arr[1])]
        if arr[2] == '?':
            return False
        # global active power in kilowatts to Amps to Ah
        ret.append(self.__convert_watts_to_amps(float(arr[2]) * (1000 / 60), float(arr[4])))
        # global reactive power in kilowatts to Amps to Ah
        ret.append(self.__convert_watts_to_amps(float(arr[3]) * (1000 / 60), float(arr[4])))
        # volts
        ret.append(float(arr[4]))
        # amps
        ret.append(float(arr[5]))
        # Sub meters from watt hours to Amp hours
        ret.append(self.__convert_Wh_to_Ah(float(arr[6]), float(arr[4])))
        ret.append(self.__convert_Wh_to_Ah(float(arr[7]), float(arr[4])))
        ret.append(self.__convert_Wh_to_Ah(float(arr[8]), float(arr[4])))

        # Active power in Ah not measured by the sub meters
        ret.append((ret[1]) - (ret[5] + ret[6] + ret[7]))

        # Power in Ah difference between volts * current and global active power
        # (volts * amps) - global active kilowatts * 1000
        # / volts
        ret.append(self.__convert_watts_to_amps((float(arr[4]) * float(arr[5])) - (float(arr[2]) * 1000), float(arr[4])))
        ret.append(float(arr[4]) * float(arr[5]))
        return ret

    def __calc_phase_from_real(self):
        self.__convert_amps_to_watts(self.global_active_Ah_min, self.voltage)


    def process_record(self, rec, ranges):
        # self.time_date = self.__date_time_vector(rec[0], ranges[0])
        self.time_date = rec[0]
        self.global_active_Ah_min = self.__global_active_Ah_vector(rec[1], ranges[1])
        self.global_reactive_Ah_min = self.__global_reactive_Ah_vector(rec[2], ranges[2])
        self.voltage = self.__voltage_vector(rec[3], ranges[3])
        self.current = self.__current_vector(rec[4], ranges[4])
        self.sub_meters = self.__sub_meter_vector([rec[5], rec[6], rec[7]], [ranges[5], ranges[6], ranges[7]])
        self.residual_active_energy = self.__residual_active_energy_vector(rec[8], ranges[8])
        self.error_active = self.__error_active_vector(rec[9], ranges[9])
        self.power = self.__power_vector(rec[10], ranges[10])

    def __date_time_timestamp(self, date, time):
        date = date.split("/")
        date = dt.date(int(date[2]), int(date[1]), int(date[0])).isoformat()
        date = dt.datetime.fromisoformat("{}T{}".format(date, time))
        return date.timestamp()  # 100,000 records processed in 0.4 seconds

    def __date_time_vector(self, val, time_date_range):
        if time_date_range[0] > val or val > time_date_range[2]:
            raise Exception("Value out of range")
        val = (val - time_date_range[0]) / (time_date_range[2] - time_date_range[0])
        return abs(val)

    def __global_active_Ah_vector(self, val, global_active_Ah_min_range):
        if global_active_Ah_min_range[0] > val or val > global_active_Ah_min_range[2]:
            raise Exception("Value out of range")
        val = (val - global_active_Ah_min_range[0]) / (global_active_Ah_min_range[2] - global_active_Ah_min_range[0])
        return abs(val)

    def __global_reactive_Ah_vector(self, val, global_reactive_Ah_min_range):
        if global_reactive_Ah_min_range[0] > val or val > global_reactive_Ah_min_range[2]:
            raise Exception("Value out of range")
        val = (val - global_reactive_Ah_min_range[0]) / (global_reactive_Ah_min_range[2] - global_reactive_Ah_min_range[0])
        return abs(val)

    def __voltage_vector(self, val, voltage_range):
        if voltage_range[0] > val or val > voltage_range[2]:
            raise Exception("Value out of range")
        val = (val - voltage_range[0]) / (voltage_range[2] - voltage_range[0])
        return abs(val)

    def __current_vector(self, val, current_range):
        if current_range[0] > val or val > current_range[2]:
            raise Exception("Value out of range")
        val = (val - current_range[0]) / (current_range[2] - current_range[0])
        return abs(val)

    def __sub_meter_vector(self, vals, sub_meters_range):
        # vals[0] = self.__convert_Wh_to_Ah(vals[0], vals[-1])
        # vals[1] = self.__convert_Wh_to_Ah(vals[1], vals[-1])
        # vals[2] = self.__convert_Wh_to_Ah(vals[2], vals[-1])

        if sub_meters_range[0][0] > vals[0] or vals[0] > sub_meters_range[0][2]:
            raise Exception("Value out of range")
        vals[0] = (vals[0] - sub_meters_range[0][0]) / (sub_meters_range[0][2] - sub_meters_range[0][0])

        if sub_meters_range[1][0] > vals[1] or vals[1] > sub_meters_range[1][2]:
            raise Exception("Value out of range")
        vals[1] = (vals[1] - sub_meters_range[1][0]) / (sub_meters_range[1][2] - sub_meters_range[1][0])

        if sub_meters_range[2][0] > vals[2] or vals[2] > sub_meters_range[2][2]:
            raise Exception("Value out of range")
        vals[2] = (vals[2] - sub_meters_range[2][0]) / (sub_meters_range[2][2] - sub_meters_range[2][0])

        return vals

    def __residual_active_energy_vector(self, val, residual_active_energy_range):
        if residual_active_energy_range[0] > val or val > residual_active_energy_range[2]:
            raise Exception("Value out of range")
        val = (val - residual_active_energy_range[0]) / (residual_active_energy_range[2] - residual_active_energy_range[0])
        return abs(val)

    def __error_active_vector(self, val, error_active_range):
        if error_active_range[0] > val or val > error_active_range[2]:
            raise Exception("Value out of range")
        val = (val - error_active_range[0]) / (error_active_range[2] - error_active_range[0])
        return abs(val)

    def __power_vector(self, val, power_range):
        if power_range[0] > val or val > power_range[2]:
            raise Exception("Value out of range")
        val = (val - power_range[0]) / (power_range[2] - power_range[0])
        return abs(val)

    def __convert_watts_to_amps(self, watts, volts):
        return watts / volts

    def __convert_amps_to_watts(self, amps, volts):
        return amps * volts

    def __convert_Wh_to_Ah(self, wh, volts):
        return wh / volts


