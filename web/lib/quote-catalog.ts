import vehicleCatalogData from "@/data/vehicle-catalog.json";
import { formatTitleCase } from "@/lib/format";

export type QuoteOption = {
  value: string;
  label: string;
};

type VehicleBrandCatalog = {
  value: string;
  label: string;
  models: QuoteOption[];
};

const vehicleCatalog = vehicleCatalogData as {
  brands: VehicleBrandCatalog[];
};

export const contractOptions = [
  { value: "Mini", label: "Essentielle" },
  { value: "Median1", label: "Equilibre" },
  { value: "Median2", label: "Confort" },
  { value: "Maxi", label: "Premium" },
] as const satisfies readonly QuoteOption[];

export const paymentFrequencyOptions = [
  { value: "Yearly", label: "Annuel" },
  { value: "Biannual", label: "Semestriel" },
  { value: "Quarterly", label: "Trimestriel" },
  { value: "Monthly", label: "Mensuel" },
] as const satisfies readonly QuoteOption[];

export const paymentStatusOptions = [
  { value: "No", label: "Situation standard" },
  { value: "Yes", label: "Situation particuliere" },
] as const satisfies readonly QuoteOption[];

export const usageOptions = [
  { value: "WorkPrivate", label: "Prive et domicile-travail" },
  { value: "Retired", label: "Retraite" },
  { value: "Professional", label: "Professionnel" },
  { value: "AllTrips", label: "Tous trajets" },
] as const satisfies readonly QuoteOption[];

export const secondDriverOptions = [
  { value: "No", label: "Non, je suis le seul conducteur" },
  { value: "Yes", label: "Oui, un deuxieme conducteur est prevu" },
] as const satisfies readonly QuoteOption[];

export const genderOptions = [
  { value: "F", label: "Femme" },
  { value: "M", label: "Homme" },
] as const satisfies readonly QuoteOption[];

export const fuelOptions = [
  { value: "Diesel", label: "Diesel" },
  { value: "Gasoline", label: "Essence" },
  { value: "Hybrid", label: "Hybride" },
] as const satisfies readonly QuoteOption[];

export const vehicleTypeOptions = [
  { value: "Tourism", label: "Voiture de tourisme" },
  { value: "Commercial", label: "Usage utilitaire" },
] as const satisfies readonly QuoteOption[];

const optionsByField = {
  type_contrat: contractOptions,
  freq_paiement: paymentFrequencyOptions,
  paiement: paymentStatusOptions,
  utilisation: usageOptions,
  conducteur2: secondDriverOptions,
  sex_conducteur1: genderOptions,
  sex_conducteur2: genderOptions,
  essence_vehicule: fuelOptions,
  type_vehicule: vehicleTypeOptions,
} as const;

export const brandOptions = vehicleCatalog.brands.map((brand) => ({
  value: brand.value,
  label: formatTitleCase(brand.value),
}));

export function getModelOptions(brandValue: string) {
  return (
    vehicleCatalog.brands.find((brand) => brand.value === brandValue)?.models.map((model) => ({
      value: model.value,
      label: formatTitleCase(model.value),
    })) ?? []
  );
}

export function getOptionLabel(
  options: readonly QuoteOption[],
  value: string | null | undefined,
  fallback = "Non renseigne",
) {
  if (!value) {
    return fallback;
  }

  return options.find((option) => option.value === value)?.label ?? value;
}

export function getChoiceLabel(fieldName: keyof typeof optionsByField, value: string | null | undefined) {
  return getOptionLabel(optionsByField[fieldName], value);
}
