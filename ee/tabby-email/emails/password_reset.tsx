import {
  Button, Heading,
  Hr, Link, Section,
  Text
} from "@react-email/components";
import * as React from "react";
import RootLayout from "../components/root-layout";

interface PasswordReset {
  email?: string;
  resetLink?: string;
}

export const PasswordReset = ({
  email = "{{EMAIL}}",
  resetLink = "{{EXTERNAL_URL}}/auth/reset-password?code={{CODE}}",
}: PasswordReset) => {
  const title = `Reset your Tabby account password`;

  return (
    <RootLayout previewText={title}>
      <Heading className="text-black text-[24px] font-normal text-center p-0 mb-[30px] mx-0">
        {title}
      </Heading>
      <Text className="text-black text-[14px] leading-[24px]">
        You recently requested a password reset for your Tabby account <strong>{email}</strong>.
      </Text>
      <Section className="text-center mt-[32px] mb-[32px]">
        <Button
          className="bg-[#645740] rounded-md text-white text-sm font-semibold no-underline text-center px-5 py-3"
          href={resetLink}
        >
          Reset Password
        </Button>
      </Section>
      <Text className="text-black text-[14px] leading-[24px]">
        or copy and paste this URL into your browser:{" "}
        <Link href={resetLink} className="text-blue-600 no-underline">
          {resetLink}
        </Link>
      </Text>
      <Hr className="border border-solid border-[#eaeaea] my-[26px] mx-0 w-full" />
      <Text className="text-[#666666] text-[12px] leading-[24px]">
        This email was intended for{" "}
        <span className="text-black">{email}</span>. If you
        were not expecting this invitation, you can ignore this email.
      </Text>
    </RootLayout>
  );
};

PasswordReset.PreviewProps = {
  email: "user@tabbyml.com",
  resetLink: "http://localhost:8080/auth/reset-password?code={{CODE}}",
} as PasswordReset;

export default PasswordReset;
