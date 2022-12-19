import numpy as np
from locust import task
from locust import between
from locust import HttpUser

sample = {
  "id": 2131,
  "year_birth": 1959,
  "education": "PhD",
  "marital_status": "Divorced",
  "income": 62859.0,
  "kidhome": 0,
  "teenhome": 1,
  "dt_customer": "2012-12-30",
  "recency": 37,
  "mntwines": 1063,
  "mntfruits": 89,
  "mntmeatproducts": 102,
  "mntfishproducts": 16,
  "mntsweetproducts": 12,
  "mntgoldprods": 25,
  "numdealspurchases": 4,
  "numwebpurchases": 9,
  "numcatalogpurchases": 4,
  "numstorepurchases": 6,
  "numwebvisitsmonth": 6,
  "acceptedcmp3": 0,
  "acceptedcmp4": 0,
  "acceptedcmp5": 0,
  "acceptedcmp1": 0,
  "acceptedcmp2": 0,
  "complain": 0,
  "cust_age": 56
}

class CustMarkTestUser(HttpUser):
    """
    Usage:
        Start locust load testing client with:
            locust -H http://localhost:3000
        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    @task
    def classify(self):
        self.client.post("/classify", json=sample)

    wait_time = between(0.01, 2)

	