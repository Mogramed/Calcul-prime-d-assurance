import vehicleCatalogData from "@/data/vehicle-catalog.json";
import { formatInteger, formatTitleCase } from "@/lib/format";

export type QuoteOption = {
  value: string;
  label: string;
};

export type VehicleVariant = {
  id: string;
  label: string;
  essence_vehicule: string;
  din_vehicule: number;
  vitesse_vehicule: number;
  cylindre_vehicule: number;
  poids_vehicule: number;
  record_count: number;
};

type VehicleModelCatalog = {
  value: string;
  label: string;
  variants: VehicleVariant[];
};

type VehicleBrandCatalog = {
  value: string;
  label: string;
  models: VehicleModelCatalog[];
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
  label: brand.label || formatTitleCase(brand.value),
}));

function getBrandCatalog(brandValue: string) {
  return vehicleCatalog.brands.find((brand) => brand.value === brandValue) ?? null;
}

function getModelCatalog(brandValue: string, modelValue: string) {
  return getBrandCatalog(brandValue)?.models.find((model) => model.value === modelValue) ?? null;
}

export function getModelOptions(brandValue: string) {
  return (
    getBrandCatalog(brandValue)?.models.map((model) => ({
      value: model.value,
      label: model.label || formatTitleCase(model.value),
    })) ?? []
  );
}

export function getVehicleVariantOptions(brandValue: string, modelValue: string) {
  return (
    getModelCatalog(brandValue, modelValue)?.variants.map((variant) => ({
      value: variant.id,
      label: variant.label,
    })) ?? []
  );
}

export function getVehicleVariant(
  brandValue: string,
  modelValue: string,
  variantId: string | null | undefined,
) {
  if (!variantId) {
    return null;
  }
  return (
    getModelCatalog(brandValue, modelValue)?.variants.find((variant) => variant.id === variantId) ??
    null
  );
}

export function getVehicleVariantLabel(variantId: string | null | undefined) {
  if (!variantId) {
    return null;
  }

  for (const brand of vehicleCatalog.brands) {
    for (const model of brand.models) {
      const variant = model.variants.find((candidate) => candidate.id === variantId);
      if (variant) {
        return variant.label;
      }
    }
  }

  return null;
}

type VehicleVariantLookup = {
  marque_vehicule: string;
  modele_vehicule: string;
  essence_vehicule: string;
  din_vehicule: string | number;
  vitesse_vehicule: string | number;
  cylindre_vehicule: string | number;
  poids_vehicule: string | number;
};

function numericMatches(actual: number, candidate: string | number) {
  return String(actual) === String(candidate).trim();
}

export function findVehicleVariantId(values: VehicleVariantLookup) {
  const variants = getModelCatalog(values.marque_vehicule, values.modele_vehicule)?.variants ?? [];
  const match = variants.find(
    (variant) =>
      variant.essence_vehicule === values.essence_vehicule &&
      numericMatches(variant.din_vehicule, values.din_vehicule) &&
      numericMatches(variant.vitesse_vehicule, values.vitesse_vehicule) &&
      numericMatches(variant.cylindre_vehicule, values.cylindre_vehicule) &&
      numericMatches(variant.poids_vehicule, values.poids_vehicule),
  );

  return match?.id ?? "";
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

export function formatVehicleMetric(value: number | string) {
  return formatInteger(Number(value));
}
