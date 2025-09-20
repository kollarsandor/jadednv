# =======================================================================

const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;

// C imports for libcurl
const c = @cImport({
    @cInclude("curl/curl.h");
});

// Error types
pub const CerebrasError = error{
    CurlInitFailed,
    CurlSetOptFailed,
    CurlPerformFailed,
    InvalidApiKey,
    InvalidResponse,
    JsonParseFailed,
    OutOfMemory,
    EnvironmentVariableNotFound,
};

// Response buffer for handling HTTP responses
const ResponseBuffer = struct {
    data: ArrayList(u8),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .data = ArrayList(u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.data.deinit();
    }

    pub fn append(self: *Self, bytes: []const u8) !void {
        try self.data.appendSlice(bytes);
    }

    pub fn get_data(self: *Self) []const u8 {
        return self.data.items;
    }

    pub fn clear(self: *Self) void {
        self.data.clearRetainingCapacity();
    }
};

// SSE data structure for streaming responses
const SSEData = struct {
    event: ?[]const u8,
    data: ?[]const u8,
    id: ?[]const u8,
    retry: ?u32,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .event = null,
            .data = null,
            .id = null,
            .retry = null,
        };
    }

    pub fn is_complete(self: *const Self) bool {
        return self.data != null and std.mem.eql(u8, self.data.?, "[DONE]");
    }
};

// Chat message structure
pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,

    const Self = @This();

    pub fn init(role: []const u8, content: []const u8) Self {
        return Self{
            .role = role,
            .content = content,
        };
    }
};

// Cerebras AI client structure
pub const CerebrasAIClient = struct {
    api_key: []const u8,
    base_url: []const u8,
    model: []const u8,
    allocator: Allocator,
    curl: ?*c.CURL,

    const Self = @This();

    pub fn init(allocator: Allocator, api_key: ?[]const u8) !Self {
        // Get API key from environment if not provided
        const key = if (api_key) |k|
            k
        else
            std.os.getenv("CEREBRAS_API_KEY") orelse return CerebrasError.EnvironmentVariableNotFound;

        if (key.len == 0) {
            return CerebrasError.InvalidApiKey;
        }

        // Initialize libcurl
        if (c.curl_global_init(c.CURL_GLOBAL_DEFAULT) != 0) {
            return CerebrasError.CurlInitFailed;
        }

        const curl = c.curl_easy_init();
        if (curl == null) {
            c.curl_global_cleanup();
            return CerebrasError.CurlInitFailed;
        }

        return Self{
            .api_key = key,
            .base_url = "https://api.cerebras.ai/v1",
            .model = "gpt-oss-120b",
            .allocator = allocator,
            .curl = curl,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.curl) |curl| {
            c.curl_easy_cleanup(curl);
        }
        c.curl_global_cleanup();
    }

    // Create JSON payload for chat completions
    pub fn create_chat_payload(self: *Self, messages: []const ChatMessage, stream: bool, max_tokens: ?u32) ![]u8 {
        var json_buffer = ArrayList(u8).init(self.allocator);
        defer json_buffer.deinit();

        try json_buffer.appendSlice("{\n");

        // Model
        try json_buffer.writer().print("  \"model\": \"{s}\",\n", .{self.model});

        // Messages array
        try json_buffer.appendSlice("  \"messages\": [\n");
        var i: usize = 0;
        for (messages) |msg| {
            if (i > 0) try json_buffer.appendSlice(",\n");
            try json_buffer.writer().print("    {{\n      \"role\": \"{s}\",\n      \"content\": \"{s}\"\n    }}", .{ msg.role, msg.content });
            i += 1;
        }
        try json_buffer.appendSlice("\n  ],\n");

        // Stream parameter
        if (stream) {
            try json_buffer.appendSlice("  \"stream\": true,\n");
        }

        // Max tokens
        if (max_tokens) |tokens| {
            try json_buffer.writer().print("  \"max_tokens\": {},\n", .{tokens});
        }

        // Temperature and other parameters
        try json_buffer.appendSlice("  \"temperature\": 0.7\n");
        try json_buffer.appendSlice("}");

        return self.allocator.dupe(u8, json_buffer.items);
    }

    // Callback function for writing response data
    fn write_callback(contents: [*c]u8, size: usize, nmemb: usize, userdata: ?*anyopaque) callconv(.C) usize {
        const real_size = size * nmemb;
        const buffer = @as(*ResponseBuffer, @ptrCast(@alignCast(userdata)));

        const data_slice = contents[0..real_size];
        buffer.append(data_slice) catch return 0;

        return real_size;
    }

    // Callback function for handling streaming data (SSE)
    fn stream_callback(contents: [*c]u8, size: usize, nmemb: usize, userdata: ?*anyopaque) callconv(.C) usize {
        const real_size = size * nmemb;
        const buffer = @as(*ResponseBuffer, @ptrCast(@alignCast(userdata)));

        const data_slice = contents[0..real_size];

        // Parse SSE data
        var lines = std.mem.split(u8, data_slice, "\n");
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "data: ")) {
                const data_content = line[6..];

                // Skip empty data lines
                if (data_content.len == 0) continue;

                // Check for [DONE] message
                if (std.mem.eql(u8, data_content, "[DONE]")) {
                    print("Stream completed.\n", .{});
                    continue;
                }

                // Try to parse JSON data
                if (parse_streaming_response(buffer.allocator, data_content)) |response| {
                    if (response.content) |content| {
                        print("{s}", .{content});
                    }
                } else |_| {
                    // Ignore parse errors and continue
                }
            }
        }

        buffer.append(data_slice) catch return 0;
        return real_size;
    }

    // Parse streaming JSON response
    fn parse_streaming_response(_: Allocator, json_data: []const u8) !struct { content: ?[]const u8 } {
        // Simple JSON parsing for streaming response
        // Look for "content" field in the delta object

        if (std.mem.indexOf(u8, json_data, "\"content\":")) |start| {
            const content_start = start + 11; // length of "\"content\":" + 1

            if (content_start < json_data.len and json_data[content_start] == '"') {
                // Find the end quote
                if (std.mem.indexOf(u8, json_data[content_start + 1..], "\"")) |end| {
                    const content = json_data[content_start + 1..content_start + 1 + end];
                    return .{ .content = content };
                }
            }
        }

        return .{ .content = null };
    }

    // Make HTTP request to Cerebras API
    pub fn make_request(self: *Self, endpoint: []const u8, payload: []const u8, stream: bool) ![]u8 {
        const curl = self.curl orelse return CerebrasError.CurlInitFailed;

        // Construct full URL
        const full_url = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.base_url, endpoint });
        defer self.allocator.free(full_url);

        // Prepare response buffer
        var response_buffer = ResponseBuffer.init(self.allocator);
        defer response_buffer.deinit();

        // Reset curl options
        c.curl_easy_reset(curl);

        // Set URL
        if (c.curl_easy_setopt(curl, c.CURLOPT_URL, full_url.ptr) != c.CURLE_OK) {
            return CerebrasError.CurlSetOptFailed;
        }

        // Set POST method
        if (c.curl_easy_setopt(curl, c.CURLOPT_POST, @as(c_long, 1)) != c.CURLE_OK) {
            return CerebrasError.CurlSetOptFailed;
        }

        // Set POST data
        if (c.curl_easy_setopt(curl, c.CURLOPT_POSTFIELDS, payload.ptr) != c.CURLE_OK) {
            return CerebrasError.CurlSetOptFailed;
        }

        if (c.curl_easy_setopt(curl, c.CURLOPT_POSTFIELDSIZE, @as(c_long, @intCast(payload.len))) != c.CURLE_OK) {
            return CerebrasError.CurlSetOptFailed;
        }

        // Prepare headers
        const auth_header = try std.fmt.allocPrint(self.allocator, "Authorization: Bearer {s}", .{self.api_key});
        defer self.allocator.free(auth_header);

        var headers: ?*c.curl_slist = null;
        headers = c.curl_slist_append(headers, "Content-Type: application/json");
        headers = c.curl_slist_append(headers, auth_header.ptr);
        if (stream) {
            headers = c.curl_slist_append(headers, "Accept: text/event-stream");
            headers = c.curl_slist_append(headers, "Cache-Control: no-cache");
        }
        defer c.curl_slist_free_all(headers);

        if (c.curl_easy_setopt(curl, c.CURLOPT_HTTPHEADER, headers) != c.CURLE_OK) {
            return CerebrasError.CurlSetOptFailed;
        }

        // Set write callback based on streaming mode
        if (stream) {
            if (c.curl_easy_setopt(curl, c.CURLOPT_WRITEFUNCTION, stream_callback) != c.CURLE_OK) {
                return CerebrasError.CurlSetOptFailed;
            }
        } else {
            if (c.curl_easy_setopt(curl, c.CURLOPT_WRITEFUNCTION, write_callback) != c.CURLE_OK) {
                return CerebrasError.CurlSetOptFailed;
            }
        }

        if (c.curl_easy_setopt(curl, c.CURLOPT_WRITEDATA, &response_buffer) != c.CURLE_OK) {
            return CerebrasError.CurlSetOptFailed;
        }

        // Enable verbose output for debugging
        // if (c.curl_easy_setopt(curl, c.CURLOPT_VERBOSE, @as(c_long, 1)) != c.CURLE_OK) {
        //     return CerebrasError.CurlSetOptFailed;
        // }

        // Perform the request
        const res = c.curl_easy_perform(curl);
        if (res != c.CURLE_OK) {
            print("CURL perform failed: {}\n", .{res});
            return CerebrasError.CurlPerformFailed;
        }

        // Check HTTP response code
        var response_code: c_long = 0;
        if (c.curl_easy_getinfo(curl, c.CURLINFO_RESPONSE_CODE, &response_code) != c.CURLE_OK) {
            return CerebrasError.CurlPerformFailed;
        }

        if (response_code < 200 or response_code >= 300) {
            print("HTTP error: {}\n", .{response_code});
            print("Response: {s}\n", .{response_buffer.get_data()});
            return CerebrasError.InvalidResponse;
        }

        return self.allocator.dupe(u8, response_buffer.get_data());
    }

    // Create chat completion with streaming support
    pub fn create_chat_completion(self: *Self, messages: []const ChatMessage, stream: bool, max_tokens: ?u32) ![]u8 {
        const payload = try self.create_chat_payload(messages, stream, max_tokens);
        defer self.allocator.free(payload);

        print("Sending request to Cerebras AI API...\n", .{});
        if (stream) {
            print("Streaming response:\n", .{});
        }

        return self.make_request("chat/completions", payload, stream);
    }

    // Helper function to create a simple chat message
    pub fn create_single_message_chat(self: *Self, user_message: []const u8, stream: bool) ![]u8 {
        const messages = [_]ChatMessage{
            ChatMessage.init("user", user_message),
        };

        return self.create_chat_completion(&messages, stream, 1000);
    }
};

// Utility function to get environment variable with fallback
fn get_env_var(key: []const u8, fallback: []const u8) []const u8 {
    return std.os.getenv(key) orelse fallback;
}

// Main function demonstrating usage
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("Cerebras AI Client Demo\n", .{});
    print("======================\n\n", .{});

    // Initialize client (will get API key from CEREBRAS_API_KEY environment variable)
    var client = CerebrasAIClient.init(allocator, null) catch |err| {
        switch (err) {
            CerebrasError.EnvironmentVariableNotFound => {
                print("Error: CEREBRAS_API_KEY environment variable not found.\n", .{});
                print("Please set your Cerebras AI API key: export CEREBRAS_API_KEY=your_api_key_here\n", .{});
                return;
            },
            CerebrasError.InvalidApiKey => {
                print("Error: Invalid API key provided.\n", .{});
                return;
            },
            else => {
                print("Error initializing client: {}\n", .{err});
                return;
            }
        }
    };
    defer client.deinit();

    print("Client initialized successfully!\n", .{});
    print("Model: {s}\n", .{client.model});
    print("Base URL: {s}\n\n", .{client.base_url});

    // Test 1: Simple non-streaming request
    print("Test 1: Non-streaming chat completion\n", .{});
    print("--------------------------------------\n", .{});

    const simple_response = client.create_single_message_chat(
        "Explain quantum computing in simple terms",
        false
    ) catch |err| {
        print("Error in simple chat: {}\n", .{err});
        return;
    };
    defer allocator.free(simple_response);

    print("Response received ({} bytes)\n", .{simple_response.len});
    if (simple_response.len < 1000) {
        print("Response: {s}\n", .{simple_response});
    } else {
        print("Response (first 500 chars): {s}...\n", .{simple_response[0..500]});
    }
    print("\n", .{});

    // Test 2: Streaming request
    print("Test 2: Streaming chat completion\n", .{});
    print("---------------------------------\n", .{});

    const stream_response = client.create_single_message_chat(
        "Write a short poem about artificial intelligence",
        true
    ) catch |err| {
        print("Error in streaming chat: {}\n", .{err});
        return;
    };
    defer allocator.free(stream_response);

    print("\n\nStreaming completed. Full response data ({} bytes received)\n", .{stream_response.len});

    // Test 3: Multi-message conversation
    print("\nTest 3: Multi-message conversation\n", .{});
    print("----------------------------------\n", .{});

    const conversation = [_]ChatMessage{
        ChatMessage.init("system", "You are a helpful AI assistant specialized in technology."),
        ChatMessage.init("user", "What is the difference between machine learning and deep learning?"),
        ChatMessage.init("assistant", "Machine learning is a broader field that includes various algorithms to learn from data, while deep learning is a subset that uses neural networks with multiple layers."),
        ChatMessage.init("user", "Can you give me a practical example?"),
    };

    const conversation_response = client.create_chat_completion(&conversation, false, 500) catch |err| {
        print("Error in conversation: {}\n", .{err});
        return;
    };
    defer allocator.free(conversation_response);

    print("Conversation response received ({} bytes)\n", .{conversation_response.len});
    if (conversation_response.len < 800) {
        print("Response: {s}\n", .{conversation_response});
    } else {
        print("Response (first 400 chars): {s}...\n", .{conversation_response[0..400]});
    }

    print("\nDemo completed successfully!\n", .{});
}

// Test function for development
test "basic client initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test with mock API key
    const mock_key = "test_api_key_12345";
    var client = try CerebrasAIClient.init(allocator, mock_key);
    defer client.deinit();

    try testing.expect(std.mem.eql(u8, client.api_key, mock_key));
    try testing.expect(std.mem.eql(u8, client.model, "gpt-oss-120b"));
    try testing.expect(std.mem.eql(u8, client.base_url, "https://api.cerebras.ai/v1"));
}

test "json payload creation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var client = try CerebrasAIClient.init(allocator, "test_key");
    defer client.deinit();

    const messages = [_]ChatMessage{
        ChatMessage.init("user", "Hello, world!"),
    };

    const payload = try client.create_chat_payload(&messages, false, 100);
    defer allocator.free(payload);

    try testing.expect(std.mem.indexOf(u8, payload, "\"model\": \"gpt-oss-120b\"") != null);
    try testing.expect(std.mem.indexOf(u8, payload, "\"role\": \"user\"") != null);
    try testing.expect(std.mem.indexOf(u8, payload, "\"content\": \"Hello, world!\"") != null);
    try testing.expect(std.mem.indexOf(u8, payload, "\"max_tokens\": 100") != null);
}
# =======================================================================


# =======================================================================
