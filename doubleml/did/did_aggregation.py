class DoubleMLDIDAggregation:

    def __init__(
        self,
        aggregated_frameworks,
        overall_aggregated_framework,
        weight_masks,
    ):
        self._aggregated_frameworks = aggregated_frameworks
        self._overall_aggregated_framework = overall_aggregated_framework
        self._weight_masks = weight_masks

    def __str__(self):
        class_name = self.__class__.__name__
        header = f"================== {class_name} Object ==================\n"

        overall_summary = str(self.overall_summary())
        aggregated_effects_summary = str(self.aggregated_summary())
        res = (
            header
            + "\n------------------ Overall Aggregated Effects ------------------\n"
            + overall_summary
            + "\n------------------ Aggregated Effects         ------------------\n"
            + aggregated_effects_summary
        )
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

    def aggregated_summary(self):
        """
        A summary for the aggregated effects.
        """
        return self.aggregated_frameworks.summary()

    def overall_summary(self):
        """
        A summary for the overall aggregated effect.
        """
        return self.overall_aggregated_framework.summary()
