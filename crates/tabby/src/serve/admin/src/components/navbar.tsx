import type { FC } from "react";
import { Button, DarkThemeToggle, Navbar } from "flowbite-react";

const ExampleNavbar: FC = function () {
  return (
    <Navbar fluid>
      <div className="w-full p-3 lg:px-5 lg:pl-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Navbar.Brand href="/">
              <img alt="" src="/images/logo.svg" className="mr-3 h-6 sm:h-8" />
              <span className="self-center whitespace-nowrap text-2xl font-semibold dark:text-white">
                Flowbite
              </span>
            </Navbar.Brand>
          </div>
          <div className="flex items-center gap-3">
            <iframe
              height="30"
              src="https://ghbtns.com/github-btn.html?user=themesberg&repo=flowbite-react-admin-dashboard&type=star&count=true&size=large"
              title="GitHub"
              width="90"
              className="hidden sm:block"
            />
            <Button color="primary" href="https://flowbite.com/pro/">
              Upgrade to Pro
            </Button>
            <DarkThemeToggle />
          </div>
        </div>
      </div>
    </Navbar>
  );
};

export default ExampleNavbar;
