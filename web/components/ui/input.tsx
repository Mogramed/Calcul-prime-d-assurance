import * as React from "react";

import { cn } from "@/lib/cn";

export function Input({
  className,
  onKeyDown,
  type,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      type={type}
      className={cn(
        "h-12 w-full rounded-2xl border border-[var(--line)] bg-white px-4 text-sm text-[var(--foreground)] outline-none transition placeholder:text-[var(--muted)] focus:border-[var(--accent)] focus:ring-2 focus:ring-[color:color-mix(in_srgb,var(--accent)_20%,transparent)]",
        className,
      )}
      onKeyDown={(event) => {
        if (type === "number" && ["e", "E", "+", "-"].includes(event.key)) {
          event.preventDefault();
        }
        onKeyDown?.(event);
      }}
      {...props}
    />
  );
}
