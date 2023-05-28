/* eslint-disable jsx-a11y/anchor-is-valid */
import { Button, Card, Checkbox, Label, TextInput } from "flowbite-react";
import type { FC } from "react";

const SignUpPage: FC = function () {
  return (
    <div className="flex flex-col items-center justify-center px-6 lg:h-screen lg:gap-y-12">
      <div className="my-6 flex items-center gap-x-1 lg:my-0">
        <img
          alt="Flowbite logo"
          src="https://flowbite.com/docs/images/logo.svg"
          className="mr-3 h-12"
        />
        <span className="self-center whitespace-nowrap text-2xl font-semibold dark:text-white">
          Flowbite
        </span>
      </div>
      <Card
        horizontal
        imgSrc="/images/authentication/create-account.jpg"
        imgAlt=""
        className="w-full md:max-w-screen-sm [&>img]:hidden md:[&>img]:w-96 md:[&>img]:p-0 md:[&>*]:w-full md:[&>*]:p-16 lg:[&>img]:block"
      >
        <h1 className="mb-3 text-2xl font-bold dark:text-white md:text-3xl">
          Create a Free Account
        </h1>
        <form>
          <div className="mb-4 flex flex-col gap-y-3">
            <Label htmlFor="email">Your email</Label>
            <TextInput
              id="email"
              name="email"
              placeholder="name@company.com"
              type="email"
            />
          </div>
          <div className="mb-6 flex flex-col gap-y-3">
            <Label htmlFor="password">Your password</Label>
            <TextInput
              id="password"
              name="password"
              placeholder="••••••••"
              type="password"
            />
          </div>
          <div className="mb-6 flex flex-col gap-y-3">
            <Label htmlFor="confirmPassword">Confirm password</Label>
            <TextInput
              id="confirmPassword"
              name="confirmPassword"
              placeholder="••••••••"
              type="password"
            />
          </div>
          <div className="mb-6 flex items-center gap-x-3">
            <Checkbox id="acceptTerms" name="acceptTerms" />
            <Label htmlFor="acceptTerms">
              I accept the&nbsp;
              <a href="#" className="text-primary-700 dark:text-primary-200">
                Terms and Conditions
              </a>
            </Label>
          </div>
          <div className="mb-7">
            <Button type="submit" className="w-full lg:w-auto">
              Create account
            </Button>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-300">
            Already have an account?&nbsp;
            <a href="#" className="text-primary-600 dark:text-primary-200">
              Login here
            </a>
          </p>
        </form>
      </Card>
    </div>
  );
};

export default SignUpPage;
