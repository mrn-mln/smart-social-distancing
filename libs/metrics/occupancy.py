import csv
import numpy as np
import os

from collections import deque
from datetime import datetime
from statistics import mean
from typing import Dict, Iterator

from .base import BaseMetric


class OccupancyMetric(BaseMetric):

    reports_folder = "occupancy"
    csv_headers = ["AverageOccupancy", "MaxOccupancy"]
    entity = "area"
    live_csv_headers = ["AverageOccupancy", "MaxOccupancy", "OccupancyThreshold", "Violations"]

    @classmethod
    def procces_csv_row(cls, csv_row: Dict, objects_logs: Dict):
        row_time = datetime.strptime(csv_row["Timestamp"], "%Y-%m-%d %H:%M:%S")
        row_hour = row_time.hour
        if not objects_logs.get(row_hour):
            objects_logs[row_hour] = {}
        if not objects_logs[row_hour].get("Occupancy"):
            objects_logs[row_hour]["Occupancy"] = []
        objects_logs[row_hour]["Occupancy"].append(int(csv_row["Occupancy"]))

    @classmethod
    def generate_hourly_metric_data(cls, objects_logs):
        summary = np.zeros((len(objects_logs), 2), dtype=np.long)
        for index, hour in enumerate(sorted(objects_logs)):
            summary[index] = (
                mean(objects_logs[hour].get("Occupancy", [0])), max(objects_logs[hour].get("Occupancy", [0]))
            )
        return summary

    @classmethod
    def generate_daily_csv_data(cls, yesterday_hourly_file):
        average_ocupancy = []
        max_occupancy = []
        with open(yesterday_hourly_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if int(row["AverageOccupancy"]):
                    average_ocupancy.append(int(row["AverageOccupancy"]))
                max_occupancy.append(int(row["MaxOccupancy"]))
        if not average_ocupancy:
            return 0, 0
        return round(mean(average_ocupancy), 2), max(max_occupancy)

    @classmethod
    def generate_live_csv_data(cls, today_entity_csv, entity, entries_in_interval):
        """
        Generates the live report using the `today_entity_csv` file received.
        """
        with open(today_entity_csv, "r") as log:
            objects_logs = {}
            lastest_entries = deque(csv.DictReader(log), entries_in_interval)
            for entry in lastest_entries:
                cls.procces_csv_row(entry, objects_logs)
            # Put the rows in the same hour
            objects_logs_merged = {
                0: {"Occupancy": []}
            }
            for hour in objects_logs:
                objects_logs_merged[0]["Occupancy"].extend(objects_logs[hour]["Occupancy"])
        occupancy_live = cls.generate_hourly_metric_data(objects_logs_merged)[0].tolist()
        occupancy_live.append(int(entity["occupancy_threshold"]))
        daily_violations = 0
        entity_directory = entity["base_directory"]
        reports_directory = os.path.join(entity_directory, "reports", cls.reports_folder)
        file_path = os.path.join(reports_directory, "live.csv")
        if os.path.exists(file_path):
            with open(file_path, "r") as live_file:
                lastest_entry = deque(csv.DictReader(live_file), 1)[0]
                if datetime.strptime(lastest_entry["Time"], "%Y-%m-%d %H:%M:%S").date() == datetime.today().date():
                    daily_violations = int(lastest_entry["Violations"])
        if occupancy_live[1] > occupancy_live[2]:
            # Max Occupancy detections > Occupancy threshold
            daily_violations += 1
        occupancy_live.append(daily_violations)
        return occupancy_live

    @classmethod
    def get_trend_live_values(cls, live_report_paths: Iterator[str]) -> Iterator[int]:
        latest_occupancy_results = {}
        for n in range(10):
            latest_occupancy_results[n] = None
        for live_path in live_report_paths:
            with open(live_path, "r") as live_file:
                lastest_10_entries = deque(csv.DictReader(live_file), 10)
                for index, item in enumerate(lastest_10_entries):
                    if not latest_occupancy_results[index]:
                        latest_occupancy_results[index] = 0
                    latest_occupancy_results[index] += int(item["MaxOccupancy"])
        return [item for item in latest_occupancy_results.values() if item is not None]
