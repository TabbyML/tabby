package com.tabbyml.codecompletion.api;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.logging.Logger;

import com.fasterxml.jackson.databind.ObjectMapper;

public class TabbyCompletionService {

	private static Logger LOG = Logger.getLogger(TabbyCompletionService.class.getName());
	private static final ObjectMapper objectMapper = new ObjectMapper();

	private String serverEndpoint;
	
	public TabbyCompletionService(String serverEndpoint) {
		super();
		this.serverEndpoint = serverEndpoint;
	}

	public String getServerEndpoint() {
		return serverEndpoint;
	}

	public void setServerEndpoint(String serverEndpoint) {
		this.serverEndpoint = serverEndpoint;
	}
	
	public TabbyV1HealthState getHealth() {
		HttpURLConnection con;
		try {
			URL obj = new URL(serverEndpoint + "/v1/health");
			con = (HttpURLConnection) obj.openConnection();
			con.setRequestMethod("GET");
	        con.setRequestProperty("Content-Type", "application/json");
            
            // Read and print the response content
            if (con.getResponseCode() == HttpURLConnection.HTTP_OK) {
                return objectMapper.readValue(con.getInputStream(), TabbyV1HealthState.class);
               
            } else {
                throw new TabbyRequestException("HTTP Health request failed with code: " + con.getResponseCode());
            }
		} catch (MalformedURLException e) {
			throw new TabbyRequestException(e);
		} catch (IOException e) {
			throw new TabbyRequestException(e);
		}
	}

	public TabbyV1CompletionResponse completeWithTabby(String prefix, String suffix) {
		try {
			URL obj = new URL(serverEndpoint + "/v1/completions");

            HttpURLConnection con = (HttpURLConnection) obj.openConnection();
            con.setRequestMethod("POST");
            con.setDoInput(true);
            con.setDoOutput(true);
            con.setRequestProperty("Content-Type", "application/json");
            
            TabbyV1CompletionRequest completionRequest = new TabbyV1CompletionRequest(prefix, suffix);

            // Encode the JSON data as bytes
            byte[] jsonDataBytes = objectMapper.writeValueAsBytes(completionRequest);
            
            // Set the content length
            con.setRequestProperty("Content-Length", String.valueOf(jsonDataBytes.length));

            // Open an output stream and send the JSON data
            try (DataOutputStream out = new DataOutputStream(con.getOutputStream())) {
                out.write(jsonDataBytes);
                out.flush();
            }
            
            int responseCode = con.getResponseCode();
            LOG.info("Tabby responsecode: " + String.valueOf(responseCode));
            
            // Read and print the response content
            if (responseCode == HttpURLConnection.HTTP_OK) {
                BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
                String inputLine;
                StringBuilder response = new StringBuilder();

                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                in.close();

                LOG.info("Response Content:\n" + response.toString());

                TabbyV1CompletionResponse completionResponse = objectMapper.readValue(response.toString(), TabbyV1CompletionResponse.class);
                
               
                return completionResponse;
            } else {
                LOG.warning("HTTP POST request failed.");
            }
		} catch (MalformedURLException e) {
			throw new TabbyRequestException(e);
		} catch (IOException e) {
			throw new TabbyRequestException(e);
		}
		return null;
	}
}
