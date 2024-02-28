import {
  Body,
  Button,
  Container,
  Column,
  Head,
  Heading,
  Hr,
  Html,
  Img,
  Link,
  Preview,
  Row,
  Section,
  Text,
} from "@react-email/components";
import { Tailwind } from "@react-email/tailwind";
import * as React from "react";

interface RootLayoutProps {
  previewText: string
  children: React.ReactNode
}

export const RootLayout = ({
  previewText,
  children,
}: RootLayoutProps) => {
  return (
    <Html>
      <Head />
      <Preview>{previewText}</Preview>
      <Tailwind>
        <Body className="bg-[#FBF9F5] my-auto mx-auto font-sans px-2">
          <Container className="bg-[#E8E2D2] border border-solid border-[#eaeaea] rounded my-[40px] mx-auto p-[20px] max-w-[465px]">
            {children}
          </Container>
        </Body>
      </Tailwind>
    </Html>
  );
};

export default RootLayout;
