import {
  Button, Heading,
  Hr, Link, Section,
  Text
} from "@react-email/components";
import * as React from "react";
import RootLayout from "../components/root-layout";

interface Invitation {
  email?: string;
  inviteLink?: string;
}

export const Invitation = ({
  email = "{{EMAIL}}",
  inviteLink = "{{EXTERNAL_URL}}/auth/signup?invitationCode={{CODE}}",
}: Invitation) => {
  const title = `You've been invited to join a Tabby server!`;

  return (
    <RootLayout previewText={title}>
      <Heading className="text-black text-[24px] font-normal text-center p-0 mb-[30px] mx-0">
        {title}
      </Heading>
      <Text className="text-black text-[14px] leading-[24px]">
        Hello,
      </Text>
      <Text className="text-black text-[14px] leading-[24px]">
        You have been invited to join a <strong>Tabby</strong> Server, where you can tap into AI-driven code completions and chat assistants.
      </Text>
      <Section className="text-center mt-[32px] mb-[32px]">
        <Button
          className="bg-[#645740] rounded-md text-white text-sm font-semibold no-underline text-center px-5 py-3"
          href={inviteLink}
        >
          Accept Invitation
        </Button>
      </Section>
      <Text className="text-black text-[14px] leading-[24px]">
        or copy and paste this URL into your browser:{" "}
        <Link href={inviteLink} className="text-blue-600 no-underline">
          {inviteLink}
        </Link>
      </Text>
      <Hr className="border border-solid border-[#eaeaea] my-[26px] mx-0 w-full" />
      <Text className="text-[#666666] text-[12px] leading-[24px]">
        This invitation was intended for{" "}
        <span className="text-black">{email}</span>. If you
        were not expecting this invitation, you can ignore this email.
      </Text>
    </RootLayout>
  );
};

Invitation.PreviewProps = {
  email: "user@tabbyml.com",
  inviteLink: "http://localhost:8080/auth/signup?invitationCode={{CODE}}",
} as Invitation;

export default Invitation;
