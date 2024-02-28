import {
  Heading, Text
} from "@react-email/components";
import * as React from "react";
import RootLayout from "../components/root-layout";

export const Test = () => {
  const title = `Your mail server is ready to go!`;

  return (
    <RootLayout previewText={title}>
      <Heading className="text-black text-[24px] font-normal text-center p-0 mb-[30px] mx-0">
        {title}
      </Heading>
      <Text className="text-black text-[14px] leading-[24px]">
        This is a test email from Tabby to confirm that your mail server configuration is correct.
      </Text>
      <Text className="text-black text-[14px] leading-[24px]">
        If you have received this email, it means that your configuration was successful. Thank you for using Tabby!
      </Text>
    </RootLayout>
  );
};

export default Test;
