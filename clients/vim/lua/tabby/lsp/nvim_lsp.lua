local M = {}

function M.setup()
    local status, lspconfig = pcall(require, "lspconfig")
    if not status then
        return false
    end
    local lspconfig_configs = require("lspconfig.configs")

    if not lspconfig_configs.tabby then
        lspconfig_configs.tabby = {
            default_config = {
                name = "tabby",
                filetypes = {"*"},
                cmd = vim.g.tabby_agent_start_command,
                single_file_support = true,
                init_options = {
                    clientCapabilities = {
                        textDocument = {
                            inlineCompletion = true
                        }
                    }
                },
                root_dir = lspconfig.util.find_git_ancestor,
                on_attach = function(client, buf)
                    vim.api.nvim_command("doautocmd <nomodeline> User tabby_lsp_on_buffer_attached")
                end
            }
        }
    end
    lspconfig.tabby.setup({})
    return true
end

function get_client()
    return vim.lsp.get_clients({
        name = "tabby"
    })[1]
end

function M.cancel_request(id)
    if id > 0 then
        local client = get_client()
        if client ~= nil then
            client.cancel_request(id)
        end
    end
end

function M.request_inline_completion(params)
    local inline_completion_params = vim.lsp.util.make_position_params()
    inline_completion_params.context = {}
    inline_completion_params.context.triggerKind = params.trigger_kind

    local client = get_client()
    if client ~= nil then
        local request_id
        _, request_id = client.request("textDocument/inlineCompletion", inline_completion_params, function(_, result)
            vim.fn["tabby#lsp#nvim_lsp#CallInlineCompletionCallback"](request_id, result)
        end)
        return request_id
    else
        return 0
    end
end

function M.notify_event(params)
    local client = get_client()
    if client ~= nil then
        client.notify("tabby/telemetry/event", params)
    end
end

return M
