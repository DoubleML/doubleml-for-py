class DoubleMLDIDAggregation:

    def __init__(
        self,
        aggregated_frameworks,
        overall_aggregated_framework,
        weight_masks,
        additional_information=None,
    ):
        self._aggregated_frameworks = aggregated_frameworks
        self._overall_aggregated_framework = overall_aggregated_framework
        self._weight_masks = weight_masks
        self._additional_information = additional_information

    def __str__(self):
        class_name = self.__class__.__name__
        header = f"================== {class_name} Object ==================\n"
        overall_summary = self.overall_summary.to_string(index=False)
        aggregated_effects_summary = self.aggregated_summary.to_string(index=True)

        res = (
            header
            + "\n------------------ Overall Aggregated Effects ------------------\n"
            + overall_summary
            + "\n------------------ Aggregated Effects         ------------------\n"
            + aggregated_effects_summary
        )
        if self.additional_information is not None:
            res += "\n------------------ Additional Information     ------------------\n"
            res += self.additional_information

        return res

    @property
    def aggregated_frameworks(self):
        """Aggregated frameworks"""
        return self._aggregated_frameworks

    @property
    def overall_aggregated_framework(self):
        """Overall aggregated framework"""
        return self._overall_aggregated_framework

    @property
    def weight_masks(self):
        """Weight masks"""
        return self._weight_masks

    @property
    def aggregated_summary(self):
        """
        A summary for the aggregated effects.
        """
        return self.aggregated_frameworks.summary

    @property
    def overall_summary(self):
        """
        A summary for the overall aggregated effect.
        """
        return self.overall_aggregated_framework.summary

    @property
    def additional_information(self):
        """Additional information"""
        if self._additional_information is None:
            add_info = None
        else:
            add_info = str()
            for key, value in self._additional_information.items():
                add_info += f"{key}: {value}\n"
        return add_info
