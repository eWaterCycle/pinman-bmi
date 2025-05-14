from typing import Any, Tuple
import numpy as np
from leakybucket import utils
from leakybucket.lumped_bmi import LumpedBmi


class PitmanBmi(LumpedBmi):
    """Demonstration of a minimal hydrological model.

    🌧️
    🪣
    💧
    """

    def initialize(self, config_file: str) -> None:
        # The model config contains the precipitation file, and the model parameter.
        self.config: dict[str, Any] = utils.read_config(config_file)

        # Get the input data:
        #   Note: the precipitation input is in [kg m-2 s-1], temperature in [K]
        self.precipitation = utils.load_var(self.config["precipitation_file"], "pr")
        
        # Temperature is not used in this implementation, but is available if you
        #    want to write your own model. You can load it by uncommenting the next line
        # self.temperature = utils.load_var(self.config["temperature_file"], "tas")

        self.time_data = self.precipitation["time"]

        # time step size in seconds (to be able to do unit conversions).
        self.timestep_size = (
            self.time_data.values[1] - self.time_data.values[0]
        ) / np.timedelta64(1, "s")

        self.current_timestep = 0
        self.end_timestep = self.time_data.size

        # Define the model states:
        self.storage = 0  # [kg m-2 == m-1 (water depth equivalent)]
        self.discharge = 0  # [m d-1]

        # The model parameters of the Pitman model:
        #  Interception storage [mm].
        self.PI = self.config["interception storage"]
        #  Ratio of impervious to total area [-].
        self.AI = self.config["ratio of impervious to total area"]
        #  Minimum catchment absorption rate [mm/month].
        self.ZMIN = self.config["minimum catchment absorption rate"]
        #  Maximum catchment absorption rate [mm/month].
        self.ZMAX = self.config["maximum catchment absorption rate"]
        #  Maximum moisture storage capacity [mm].
        self.ST = self.config["maximum moisture storage capacity"]
        #  Moisture storage capacity below which no runoff occurs [mm].
        self.SL = self.config["moisture storage capacity below which no runoff occurs"]
        #  Runoff from moisture storage at full capacity [mm/month].
        self.FT = self.config["runoff from moisture storage at full capacity"]
        #  Maximum groundwater runoff [mm/month].
        self.GW = self.config["maximum groundwater runoff"]
        #  Evaporation-moisture storage relationship parameter [-].
        self.R = self.config["evaporation-moisture storage relationship parameter"]
        #  Power of the moisture storage-runoff equation [-].
        self.POW = self.config["power of the moisture storage-runoff equation"]
        #  Lag for surface and soil moisture [months].
        self.TL = self.config["lag for surface and soil moisture"]
        #  Lag for groundwater runoff [months].
        self.GL = self.config["lag for groundwater runoff"]

    def update(self) -> None:
        if self.current_timestep < self.end_timestep:
            # Add the current timestep's precipitation to the storage
            self.storage += (
                self.precipitation.isel(time=self.current_timestep).to_numpy()
                * self.timestep_size
            )

            # Calculate the discharge [m d-1] based on the leakiness and storage
            self.discharge = self.storage * self.leakiness
            # Subtract this discharge from the storage
            #   The discharge in [m d-1] has to be converted to [m] per timestep.
            self.storage -= self.discharge * (self.timestep_size / 24 / 3600)

            # Advance the model time by one step
            self.current_timestep += 1

    def get_component_name(self) -> str:
        return "leakybucket"

    def get_value(self, var_name: str, dest: np.ndarray) -> np.ndarray:
        match var_name:
            case "storage":
                dest[:] = np.array(self.storage)
                return dest
            case "discharge":
                dest[:] = np.array(self.discharge / (self.timestep_size / 24 / 3600))
                return dest
            case _:
                raise ValueError(f"Unknown variable {var_name}")

    def get_var_units(self, var_name: str) -> str:
        match var_name:
            case "storage":
                return "m"
            case "discharge":
                return "m d-1"
            case _:
                raise ValueError(f"Unknown variable {var_name}")

    def set_value(self, var_name: str, src: np.ndarray) -> None:
        match var_name:
            case "storage":
                self.storage = src[0]
            case _:
                raise ValueError(f"Cannot set value of var {var_name}")

    def get_output_var_names(self) -> Tuple[str]:
        return ("discharge",)
