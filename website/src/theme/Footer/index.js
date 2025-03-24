import React from "react";
import Footer from "@theme-original/Footer";

export default function FooterWrapper(props) {
  return (
    <>
      <Footer {...props} />
      <script
        defer
        src='https://static.cloudflareinsights.com/beacon.min.js'
        data-cf-beacon='{"token": "402d07214b7d41c9bf5b15cd15643391"}'
      ></script>
    </>
  );
}
