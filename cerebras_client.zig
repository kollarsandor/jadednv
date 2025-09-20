# =======================================================================

const std = @import("std");
const print = std.debug.print;
const json = std.json;
const http = std.http;
const mem = std.mem;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

/// Cerebras AI Client for production use
/// Implements complete streaming chat completions API
/// Compatible with Cerebras Cloud API v1
pub const CerebrasClient = struct {
    const Self = @This();

    // API Configuration
    const BASE_URL = "https://api.cerebras.ai";
    const CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions";
    const USER_AGENT = "cerebras-zig-client/1.0.0";
    const MAX_RESPONSE_SIZE = 50 * 1024 * 1024; // 50MB max response
    const CHUNK_SIZE = 4096; // 4KB chunks for streaming

    // Client state
    allocator: Allocator,
    api_key: []const u8,
    http_client: http.Client,

    // Error types
    pub const CerebrasError = error{
        MissingApiKey,
        InvalidApiKey,
        NetworkError,
        JsonParseError,
        InvalidResponse,
        StreamingError,
        RateLimitExceeded,
        ModelNotFound,
        InvalidParameters,
        OutOfMemory,
        EnvironmentVariableNotFound,
        InvalidUtf8,
    } || std.http.Client.RequestError || json.ParseError(json.Scanner) || Allocator.Error;

    // Message struct for chat completions
    pub const Message = struct {
        role: []const u8, // "user", "assistant", "system"
        content: []const u8,
    };

    // Chat completion request parameters
    pub const ChatRequest = struct {
        model: []const u8 = "gpt-oss-120b",
        messages: []const Message,
        stream: bool = true,
        max_completion_tokens: u32 = 65536,
        temperature: f32 = 0.24,
        top_p: f32 = 0.18,
        reasoning_effort: []const u8 = "low",
        stop: ?[]const []const u8 = null,
    };

    // Streaming response chunk
    pub const StreamChunk = struct {
        content: ?[]const u8 = null,
        finish_reason: ?[]const u8 = null,
        role: ?[]const u8 = null,
        is_complete: bool = false,
        error_message: ?[]const u8 = null,
    };

    // Response usage statistics
    pub const Usage = struct {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
    };

    // Complete response (non-streaming)
    pub const ChatResponse = struct {
        content: []const u8,
        finish_reason: []const u8,
        usage: Usage,
        model: []const u8,
    };

    /// Initialize Cerebras client with API key from environment
    pub fn init(allocator: Allocator) CerebrasError!Self {
        const api_key = std.process.getEnvVarOwned(allocator, "CEREBRAS_API_KEY") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => {
                std.log.err("CEREBRAS_API_KEY environment variable not found", .{});
                return CerebrasError.MissingApiKey;
            },
            else => return err,
        };

        if (api_key.len == 0) {
            allocator.free(api_key);
            std.log.err("CEREBRAS_API_KEY environment variable is empty", .{});
            return CerebrasError.MissingApiKey;
        }

        var http_client = http.Client{ .allocator = allocator };

        std.log.info("Cerebras client initialized with API key (length: {})", .{api_key.len});

        return Self{
            .allocator = allocator,
            .api_key = api_key,
            .http_client = http_client,
        };
    }

    /// Cleanup resources
    pub fn deinit(self: *Self) void {
        self.http_client.deinit();
        self.allocator.free(self.api_key);
    }

    /// Create streaming chat completion request
    pub fn chatStreamingRequest(self: *Self, request: ChatRequest, callback: *const fn (StreamChunk) void) CerebrasError!void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();

        // Build request JSON
        const request_json = try buildRequestJson(arena_allocator, request);
        defer arena_allocator.free(request_json);

        std.log.info("Sending streaming request to Cerebras API", .{});
        std.log.debug("Request JSON: {s}", .{request_json});

        // Parse URI
        const uri_string = try std.fmt.allocPrint(arena_allocator, "{s}{s}", .{ BASE_URL, CHAT_COMPLETIONS_ENDPOINT });
        const uri = std.Uri.parse(uri_string) catch {
            std.log.err("Failed to parse URI: {s}", .{uri_string});
            return CerebrasError.InvalidParameters;
        };

        // Setup headers
        var headers = http.Headers{ .allocator = arena_allocator };
        defer headers.deinit();

        const auth_header = try std.fmt.allocPrint(arena_allocator, "Bearer {s}", .{self.api_key});
        try headers.append("authorization", auth_header);
        try headers.append("content-type", "application/json");
        try headers.append("accept", "text/event-stream");
        try headers.append("user-agent", USER_AGENT);
        try headers.append("cache-control", "no-cache");

        // Create HTTP request
        var http_request = self.http_client.request(.POST, uri, headers, .{}) catch |err| {
            std.log.err("Failed to create HTTP request: {}", .{err});
            return CerebrasError.NetworkError;
        };
        defer http_request.deinit();

        // Set request body
        http_request.transfer_encoding = .{ .content_length = request_json.len };

        // Start request
        http_request.start() catch |err| {
            std.log.err("Failed to start HTTP request: {}", .{err});
            return CerebrasError.NetworkError;
        };

        // Send request body
        _ = http_request.writer().writeAll(request_json) catch |err| {
            std.log.err("Failed to write request body: {}", .{err});
            return CerebrasError.NetworkError;
        };

        // Finish sending
        http_request.finish() catch |err| {
            std.log.err("Failed to finish HTTP request: {}", .{err});
            return CerebrasError.NetworkError;
        };

        // Wait for response
        http_request.wait() catch |err| {
            std.log.err("Failed to wait for HTTP response: {}", .{err});
            return CerebrasError.NetworkError;
        };

        // Check response status
        switch (http_request.response.status) {
            .ok => {},
            .unauthorized => {
                std.log.err("Unauthorized: Invalid API key", .{});
                return CerebrasError.InvalidApiKey;
            },
            .too_many_requests => {
                std.log.err("Rate limit exceeded", .{});
                return CerebrasError.RateLimitExceeded;
            },
            .not_found => {
                std.log.err("Model not found", .{});
                return CerebrasError.ModelNotFound;
            },
            else => {
                std.log.err("HTTP error: {}", .{http_request.response.status});
                return CerebrasError.InvalidResponse;
            },
        }

        std.log.info("Received streaming response, processing chunks...", .{});

        // Process streaming response
        try self.processStreamingResponse(&http_request, callback, arena_allocator);
    }

    /// Process streaming SSE response
    fn processStreamingResponse(self: *Self, http_request: *http.Client.Request, callback: *const fn (StreamChunk) void, arena_allocator: Allocator) CerebrasError!void {
        _ = self;

        var buffer: [CHUNK_SIZE]u8 = undefined;
        var line_buffer = ArrayList(u8).init(arena_allocator);
        defer line_buffer.deinit();

        var bytes_read: usize = 0;
        var total_chunks: u32 = 0;

        while (true) {
            // Read chunk from stream
            const chunk_size = http_request.reader().read(&buffer) catch |err| {
                std.log.err("Failed to read from stream: {}", .{err});
                return CerebrasError.StreamingError;
            };

            if (chunk_size == 0) {
                std.log.info("Stream ended, total chunks processed: {}", .{total_chunks});
                break; // End of stream
            }

            bytes_read += chunk_size;

            // Process each character to find complete lines
            for (buffer[0..chunk_size]) |char| {
                if (char == '\n') {
                    // Process complete line
                    const line = line_buffer.items;
                    if (line.len > 0) {
                        try processSSELine(line, callback, arena_allocator, &total_chunks);
                        line_buffer.clearRetainingCapacity();
                    }
                } else if (char != '\r') {
                    try line_buffer.append(char);
                }
            }

            // Prevent infinite loops
            if (bytes_read > MAX_RESPONSE_SIZE) {
                std.log.err("Response too large: {} bytes", .{bytes_read});
                return CerebrasError.InvalidResponse;
            }
        }

        // Process final line if any
        if (line_buffer.items.len > 0) {
            try processSSELine(line_buffer.items, callback, arena_allocator, &total_chunks);
        }

        // Send completion signal
        const final_chunk = StreamChunk{
            .content = null,
            .finish_reason = "stop",
            .is_complete = true,
        };
        callback(final_chunk);
    }

    /// Process individual SSE line
    fn processSSELine(line: []const u8, callback: *const fn (StreamChunk) void, arena_allocator: Allocator, total_chunks: *u32) CerebrasError!void {

        // Skip empty lines and comments
        if (line.len == 0 or line[0] == ':') {
            return;
        }

        // Check for data line
        if (mem.startsWith(u8, line, "data: ")) {
            const data = line[6..]; // Skip "data: "

            // Check for end of stream
            if (mem.eql(u8, data, "[DONE]")) {
                std.log.info("Received [DONE] signal", .{});
                return;
            }

            // Parse JSON chunk
            const chunk = parseStreamChunk(data, arena_allocator) catch |err| {
                std.log.warn("Failed to parse chunk: {} - Data: {s}", .{ err, data });
                return;
            };

            total_chunks.* += 1;

            if (chunk.content) |content| {
                std.log.debug("Chunk {}: {s}", .{ total_chunks.*, content });
            }

            // Call user callback
            callback(chunk);
        } else if (mem.startsWith(u8, line, "event: ")) {
            const event = line[7..]; // Skip "event: "
            std.log.debug("SSE Event: {s}", .{event});
        }
    }

    /// Parse individual stream chunk JSON with safe memory handling
    fn parseStreamChunk(data: []const u8, arena_allocator: Allocator) CerebrasError!StreamChunk {
        // Validate input data first
        if (data.len == 0) {
            return StreamChunk{};
        }

        // Validate UTF-8 to prevent segfaults
        if (!std.unicode.utf8ValidateSlice(data)) {
            std.log.warn("Invalid UTF-8 in JSON data, attempting to clean...", .{});
            // Try to clean the data by replacing invalid bytes
            var cleaned_data = ArrayList(u8).init(arena_allocator);
            defer cleaned_data.deinit();

            for (data) |byte| {
                if (byte >= 32 and byte <= 126) {
                    try cleaned_data.append(byte);
                } else if (byte == '\n' or byte == '\r' or byte == '\t') {
                    try cleaned_data.append(byte);
                } else {
                    // Replace invalid bytes with space
                    try cleaned_data.append(' ');
                }
            }

            if (cleaned_data.items.len == 0) {
                return StreamChunk{};
            }

            return parseStreamChunkSafe(cleaned_data.items, arena_allocator);
        }

        return parseStreamChunkSafe(data, arena_allocator);
    }

    /// Safe JSON parsing implementation
    fn parseStreamChunkSafe(data: []const u8, arena_allocator: Allocator) CerebrasError!StreamChunk {
        // Use a more controlled parsing approach
        var parsed = json.parseFromSlice(json.Value, arena_allocator, data, .{
            .allocate = .alloc_if_needed,
            .duplicate_field_behavior = .use_first,
            .ignore_unknown_fields = true,
            .max_value_len = 1024 * 1024, // 1MB limit
        }) catch |err| {
            std.log.err("JSON parse error: {} - Data preview: {s}", .{ err, data[0..@min(100, data.len)] });
            return CerebrasError.JsonParseError;
        };
        defer parsed.deinit();

        const root = parsed.value;
        var chunk = StreamChunk{};

        // Safely extract data with null checks and bounds validation
        if (root == .object) {
            // Extract choices array with safety checks
            if (root.object.get("choices")) |choices_value| {
                if (choices_value == .array and choices_value.array.items.len > 0) {
                    const choice = choices_value.array.items[0];

                    if (choice == .object) {
                        // Extract delta content safely
                        if (choice.object.get("delta")) |delta| {
                            if (delta == .object) {
                                if (delta.object.get("content")) |content_value| {
                                    if (content_value == .string) {
                                        const content_str = content_value.string;
                                        if (content_str.len > 0 and content_str.len < 10000) { // Reasonable size limit
                                            chunk.content = try safeStringDupe(arena_allocator, content_str);
                                        }
                                    }
                                }

                                if (delta.object.get("role")) |role_value| {
                                    if (role_value == .string) {
                                        const role_str = role_value.string;
                                        if (role_str.len > 0 and role_str.len < 50) { // Role should be short
                                            chunk.role = try safeStringDupe(arena_allocator, role_str);
                                        }
                                    }
                                }
                            }
                        }

                        // Extract finish reason safely
                        if (choice.object.get("finish_reason")) |finish_reason_value| {
                            if (finish_reason_value == .string) {
                                const reason_str = finish_reason_value.string;
                                if (reason_str.len > 0 and reason_str.len < 100) {
                                    chunk.finish_reason = try safeStringDupe(arena_allocator, reason_str);
                                }
                            }
                        }
                    }
                }
            }

            // Check for errors safely
            if (root.object.get("error")) |error_value| {
                if (error_value == .object) {
                    if (error_value.object.get("message")) |message_value| {
                        if (message_value == .string) {
                            const error_str = message_value.string;
                            if (error_str.len > 0 and error_str.len < 1000) {
                                chunk.error_message = try safeStringDupe(arena_allocator, error_str);
                            }
                        }
                    }
                }
            }
        }

        return chunk;
    }

    /// Safely duplicate string with validation
    fn safeStringDupe(allocator: Allocator, str: []const u8) ![]u8 {
        if (str.len == 0) {
            return try allocator.dupe(u8, "");
        }

        // Validate UTF-8 before duplication
        if (!std.unicode.utf8ValidateSlice(str)) {
            // Clean invalid UTF-8 characters
            var cleaned = ArrayList(u8).init(allocator);
            defer cleaned.deinit();

            for (str) |byte| {
                if (byte >= 32 and byte <= 126) {
                    try cleaned.append(byte);
                } else if (byte == '\n' or byte == '\r' or byte == '\t') {
                    try cleaned.append(byte);
                } else {
                    try cleaned.append('?'); // Replace invalid with ?
                }
            }

            return try allocator.dupe(u8, cleaned.items);
        }

        return try allocator.dupe(u8, str);
    }

    /// Build JSON request body using manual string construction
    fn buildRequestJson(arena_allocator: Allocator, request: ChatRequest) CerebrasError![]u8 {
        var json_parts = ArrayList(u8).init(arena_allocator);
        defer json_parts.deinit();

        // Start JSON object
        try json_parts.appendSlice("{");

        // Model with safe escaping
        try json_parts.appendSlice("\"model\":\"");
        try escapeJsonString(arena_allocator, &json_parts, request.model);
        try json_parts.appendSlice("\",");

        // Stream
        if (request.stream) {
            try json_parts.appendSlice("\"stream\":true,");
        } else {
            try json_parts.appendSlice("\"stream\":false,");
        }

        // Max completion tokens
        const max_tokens_str = try std.fmt.allocPrint(arena_allocator, "\"max_completion_tokens\":{},", .{request.max_completion_tokens});
        try json_parts.appendSlice(max_tokens_str);

        // Temperature
        const temp_str = try std.fmt.allocPrint(arena_allocator, "\"temperature\":{d},", .{request.temperature});
        try json_parts.appendSlice(temp_str);

        // Top P
        const top_p_str = try std.fmt.allocPrint(arena_allocator, "\"top_p\":{d},", .{request.top_p});
        try json_parts.appendSlice(top_p_str);

        // Reasoning effort with safe escaping
        try json_parts.appendSlice("\"reasoning_effort\":\"");
        try escapeJsonString(arena_allocator, &json_parts, request.reasoning_effort);
        try json_parts.appendSlice("\",");

        // Messages array
        try json_parts.appendSlice("\"messages\":[");
        for (request.messages, 0..) |message, i| {
            if (i > 0) try json_parts.appendSlice(",");

            try json_parts.appendSlice("{\"role\":\"");
            try escapeJsonString(arena_allocator, &json_parts, message.role);
            try json_parts.appendSlice("\",\"content\":\"");

            // Escape content for JSON with comprehensive safety checks
            try escapeJsonString(arena_allocator, &json_parts, message.content);

            try json_parts.appendSlice("\"}");
        }
        try json_parts.appendSlice("]");

        // Stop sequences (optional) with safe escaping
        if (request.stop) |stop_sequences| {
            try json_parts.appendSlice(",\"stop\":[");
            for (stop_sequences, 0..) |stop_sequence, i| {
                if (i > 0) try json_parts.appendSlice(",");
                try json_parts.appendSlice("\"");
                try escapeJsonString(arena_allocator, &json_parts, stop_sequence);
                try json_parts.appendSlice("\"");
            }
            try json_parts.appendSlice("]");
        }

        // End JSON object
        try json_parts.appendSlice("}");

        return try arena_allocator.dupe(u8, json_parts.items);
    }

    /// Safely escape JSON string with comprehensive character handling
    fn escapeJsonString(allocator: Allocator, json_parts: *ArrayList(u8), input: []const u8) !void {
        // First validate UTF-8 to prevent segfaults
        if (!std.unicode.utf8ValidateSlice(input)) {
            std.log.warn("Invalid UTF-8 in string, cleaning before JSON escape", .{});
            // Clean the string first
            var cleaned = ArrayList(u8).init(allocator);
            defer cleaned.deinit();

            for (input) |byte| {
                if (byte >= 32 and byte <= 126) {
                    try cleaned.append(byte);
                } else if (byte == '\n' or byte == '\r' or byte == '\t') {
                    try cleaned.append(byte);
                } else {
                    // Skip invalid bytes
                    continue;
                }
            }

            return escapeJsonStringValidated(json_parts, cleaned.items);
        }

        return escapeJsonStringValidated(json_parts, input);
    }

    /// Escape validated JSON string (assumes valid UTF-8)
    fn escapeJsonStringValidated(json_parts: *ArrayList(u8), input: []const u8) !void {
        for (input) |char| {
            switch (char) {
                '"' => try json_parts.appendSlice("\\\""),
                '\\' => try json_parts.appendSlice("\\\\"),
                '\n' => try json_parts.appendSlice("\\n"),
                '\r' => try json_parts.appendSlice("\\r"),
                '\t' => try json_parts.appendSlice("\\t"),
                8 => try json_parts.appendSlice("\\\\b"), // backspace
                12 => try json_parts.appendSlice("\\\\f"), // form feed
                0x00...0x07, 0x0B, 0x0E...0x1F => {
                    // Other control characters - escape as unicode
                    // Excludes: \t=0x09, \n=0x0A, \f=0x0C, \r=0x0D, \b=0x08
                    if (char != 8 and char != 12) { // Extra safety check
                        const unicode_escape = try std.fmt.allocPrint(json_parts.allocator, "\\u{X:0>4}", .{char});
                        defer json_parts.allocator.free(unicode_escape);
                        try json_parts.appendSlice(unicode_escape);
                    }
                },
                0x7F => try json_parts.appendSlice("\\u007F"), // DEL character
                else => {
                    // Valid printable character
                    if (char >= 32 and char <= 126) {
                        try json_parts.append(char);
                    } else {
                        // Non-ASCII but valid UTF-8 - keep as is
                        try json_parts.append(char);
                    }
                },
            }
        }
    }

    /// Non-streaming chat completion (utility method)
    /// Note: For production use, prefer streaming approach for better performance
    pub fn chatCompletion(self: *Self, request: ChatRequest) CerebrasError!ChatResponse {
        // Create a modified request for non-streaming
        var non_streaming_request = request;
        non_streaming_request.stream = false;

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();

        // Build request JSON
        const request_json = try buildRequestJson(arena_allocator, non_streaming_request);

        // Parse URI
        const uri_string = try std.fmt.allocPrint(arena_allocator, "{s}{s}", .{ BASE_URL, CHAT_COMPLETIONS_ENDPOINT });
        const uri = std.Uri.parse(uri_string) catch {
            return CerebrasError.InvalidParameters;
        };

        // Setup headers
        var headers = http.Headers{ .allocator = arena_allocator };
        defer headers.deinit();

        const auth_header = try std.fmt.allocPrint(arena_allocator, "Bearer {s}", .{self.api_key});
        try headers.append("authorization", auth_header);
        try headers.append("content-type", "application/json");
        try headers.append("accept", "application/json");
        try headers.append("user-agent", USER_AGENT);

        // Create and send request
        var http_request = self.http_client.request(.POST, uri, headers, .{}) catch {
            return CerebrasError.NetworkError;
        };
        defer http_request.deinit();

        http_request.transfer_encoding = .{ .content_length = request_json.len };

        try http_request.start();
        _ = try http_request.writer().writeAll(request_json);
        try http_request.finish();
        try http_request.wait();

        // Check response status
        switch (http_request.response.status) {
            .ok => {},
            .unauthorized => return CerebrasError.InvalidApiKey,
            .too_many_requests => return CerebrasError.RateLimitExceeded,
            .not_found => return CerebrasError.ModelNotFound,
            else => return CerebrasError.InvalidResponse,
        }

        // Read response body
        const response_body = try http_request.reader().readAllAlloc(arena_allocator, MAX_RESPONSE_SIZE);

        // Parse JSON response
        const parsed = try json.parseFromSlice(json.Value, arena_allocator, response_body, .{});
        const root = parsed.value;

        // Extract response data
        var content: []const u8 = "";
        var finish_reason: []const u8 = "stop";

        if (root.object.get("choices")) |choices_value| {
            if (choices_value.array.items.len > 0) {
                const choice = choices_value.array.items[0];

                if (choice.object.get("message")) |message| {
                    if (message.object.get("content")) |content_value| {
                        if (content_value == .string) {
                            content = try self.allocator.dupe(u8, content_value.string);
                        }
                    }
                }

                if (choice.object.get("finish_reason")) |finish_reason_value| {
                    if (finish_reason_value == .string) {
                        finish_reason = try self.allocator.dupe(u8, finish_reason_value.string);
                    }
                }
            }
        }

        // Extract usage statistics
        var usage = Usage{ .prompt_tokens = 0, .completion_tokens = 0, .total_tokens = 0 };
        if (root.object.get("usage")) |usage_value| {
            if (usage_value.object.get("prompt_tokens")) |prompt_tokens| {
                if (prompt_tokens == .integer) {
                    usage.prompt_tokens = @intCast(prompt_tokens.integer);
                }
            }
            if (usage_value.object.get("completion_tokens")) |completion_tokens| {
                if (completion_tokens == .integer) {
                    usage.completion_tokens = @intCast(completion_tokens.integer);
                }
            }
            if (usage_value.object.get("total_tokens")) |total_tokens| {
                if (total_tokens == .integer) {
                    usage.total_tokens = @intCast(total_tokens.integer);
                }
            }
        }

        return ChatResponse{
            .content = content,
            .finish_reason = finish_reason,
            .usage = usage,
            .model = request.model,
        };
    }
};

/// Example usage and testing function
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("Initializing Cerebras client...", .{});

    var client = CerebrasClient.init(allocator) catch |err| {
        std.log.err("Failed to initialize client: {}", .{err});
        return;
    };
    defer client.deinit();

    // Example messages
    const messages = [_]CerebrasClient.Message{
        .{ .role = "system", .content = "You are a helpful AI assistant." },
        .{ .role = "user", .content = "Explain quantum computing in simple terms." },
    };

    const request = CerebrasClient.ChatRequest{
        .model = "gpt-oss-120b",
        .messages = &messages,
        .stream = true,
        .max_completion_tokens = 65536,
        .temperature = 0.24,
        .top_p = 0.18,
        .reasoning_effort = "low",
    };

    std.log.info("Starting streaming chat completion...", .{});

    // Define streaming callback
    const callback = struct {
        fn handle(chunk: CerebrasClient.StreamChunk) void {
            if (chunk.error_message) |error_msg| {
                std.log.err("Stream error: {s}", .{error_msg});
                return;
            }

            if (chunk.content) |content| {
                print("{s}", .{content});
            }

            if (chunk.is_complete) {
                print("\n\n[Streaming completed]\n", .{});
            }
        }
    }.handle;

    // Execute streaming request
    client.chatStreamingRequest(request, &callback) catch |err| {
        std.log.err("Streaming request failed: {}", .{err});
        return;
    };

    std.log.info("Example completed successfully!", .{});
}

// Export for C FFI integration
export fn cerebras_create_client() ?*CerebrasClient {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const client = allocator.create(CerebrasClient) catch return null;
    client.* = CerebrasClient.init(allocator) catch {
        allocator.destroy(client);
        return null;
    };

    return client;
}

export fn cerebras_destroy_client(client: ?*CerebrasClient) void {
    if (client) |c| {
        const allocator = c.allocator;
        c.deinit();
        allocator.destroy(c);
    }
}
# =======================================================================


# =======================================================================
