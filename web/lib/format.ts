export function formatCurrency(value: number) {
  return new Intl.NumberFormat("fr-FR", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatDecimal(value: number) {
  return new Intl.NumberFormat("fr-FR", {
    maximumFractionDigits: 4,
  }).format(value);
}

export function formatInteger(value: number) {
  return new Intl.NumberFormat("fr-FR", {
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatDateTime(value: string) {
  return new Intl.DateTimeFormat("fr-FR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function formatTitleCase(value: string) {
  return value
    .split(" ")
    .filter(Boolean)
    .map((token) => {
      if (/^\d+$/.test(token)) {
        return token;
      }

      if (token === token.toUpperCase() && (token.length <= 3 || /\d/.test(token))) {
        return token;
      }

      if (/^[A-Z]\d+$/i.test(token)) {
        return token.toUpperCase();
      }

      return token.slice(0, 1) + token.slice(1).toLowerCase();
    })
    .join(" ");
}
