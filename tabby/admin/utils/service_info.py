from dataclasses import dataclass

import requests
from requests.exceptions import ConnectionError


@dataclass
class ServiceInfo:
    label: str
    health_url: str

    @property
    def is_health(self) -> bool:
        try:
            return requests.get(self.health_url).status_code == 200
        except ConnectionError:
            return False

    @property
    def badge_url(self) -> str:
        is_health = self.is_health
        label = self.label.replace("-", "--")
        message = "live" if is_health else "down"
        color = "green" if is_health else "red"

        return f"https://img.shields.io/badge/{label}-{message}-{color}"
