"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { LogOut } from "lucide-react";
import { useRouter } from "next/navigation";

import { Button } from "@/components/ui/button";
import { logoutAccount } from "@/lib/web-api";

export function LogoutButton() {
  const router = useRouter();
  const queryClient = useQueryClient();

  const logoutMutation = useMutation({
    mutationFn: logoutAccount,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["auth-session"] });
      await queryClient.invalidateQueries({ queryKey: ["quotes"] });
      router.push("/");
      router.refresh();
    },
  });

  return (
    <Button
      type="button"
      variant="secondary"
      disabled={logoutMutation.isPending}
      onClick={() => logoutMutation.mutate()}
    >
      <LogOut className="mr-2 h-4 w-4" />
      Se deconnecter
    </Button>
  );
}
