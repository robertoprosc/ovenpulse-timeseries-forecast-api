



class BakeMeasurement:
  """
  This class represents a bake measurement with fields populated from a request dictionary.
  """

  def __init__(self, request_dict):
    """
    Initializes the BakeMeasurement object with data from a request dictionary.

    Args:
      request_dict (dict): A dictionary containing keys corresponding to the measurement fields.
    """

    # Validate required fields are present in the dictionary
    required_fields = [
        "Phase_01", "Phase_02", "Phase_03", "Phase_04", "Phase_05",
        "Phase_06", "Phase_07", "Phase_08", "Phase_09", "Cake_ID", "Oven"
    ]

    missing_fields = [field for field in required_fields if field not in request_dict]
    if missing_fields:
      raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    # Assign field values based on dictionary keys
    for field_name, value in request_dict.items():
      setattr(self, field_name, value)

  # Getter methods for each field
  @property
  def phase_01(self):
    return getattr(self, "Phase_01")

  @property
  def phase_02(self):
    return getattr(self, "Phase_02")

  @property
  def phase_03(self):
    return getattr(self, "Phase_03")

  @property
  def phase_04(self):
    return getattr(self, "Phase_04")

  @property
  def phase_05(self):
    return getattr(self, "Phase_05")

  @property
  def phase_06(self):
    return getattr(self, "Phase_06")

  @property
  def phase_07(self):
    return getattr(self, "Phase_07")

  @property
  def phase_08(self):
    return getattr(self, "Phase_08")

  @property
  def phase_09(self):
    return getattr(self, "Phase_09")

  @property
  def cake_id(self):
    return getattr(self, "Cake_ID")

  @property
  def oven(self):
    return getattr(self, "Oven")

  def to_dict(self):
    """
    Returns a dictionary representation of the BakeMeasurement object.
    """
    return {
      "Phase_01": self.phase_01,
      "Phase_02": self.phase_02,
      "Phase_03": self.phase_03,
      "Phase_04": self.phase_04,
      "Phase_05": self.phase_05,
      "Phase_06": self.phase_06,
      "Phase_07": self.phase_07,
      "Phase_08": self.phase_08,
      "Phase_09": self.phase_09,
      "Cake_ID": self.cake_id,
      "Oven": self.oven
    }