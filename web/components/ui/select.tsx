import * as React from "react";

import { cn } from "@/lib/cn";

export function Select({ className, children, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        "h-12 w-full appearance-none rounded-2xl border border-[var(--line)] bg-white px-4 text-sm text-[var(--foreground)] outline-none transition focus:border-[var(--accent)] focus:ring-2 focus:ring-[color:color-mix(in_srgb,var(--accent)_20%,transparent)] disabled:cursor-not-allowed disabled:bg-[var(--surface-alt)] disabled:text-[var(--muted)]",
        className,
      )}
      {...props}
    >
      {children}
    </select>
  );
}
