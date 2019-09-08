
import json


class ResponseObj:

    def __init__(self, status, output):
        self.status = status
        self.display_text = output

    def format(self):
        result = {}
        if self.status not in ["success", "error"]:
            raise Exception("status is only success or error")
        result["status"] = self.status
        if not self.display_text:
            raise Exception("displayText cannot be None")
        result["output"] = self.display_text
        return json.dumps(result)
