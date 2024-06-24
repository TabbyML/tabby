package com.tabbyml.intellijtabby.lsp.protocol

import com.google.gson.Gson
import com.google.gson.TypeAdapter
import com.google.gson.TypeAdapterFactory
import com.google.gson.reflect.TypeToken
import com.google.gson.stream.JsonReader
import com.google.gson.stream.JsonWriter

class EnumIntTypeAdapter<T : EnumInt>(private val type: Class<T>) : TypeAdapter<T>() {
  override fun write(writer: JsonWriter, value: T?) {
    if (value == null) {
      writer.nullValue()
      return
    }
    writer.value(value.value)
  }

  override fun read(reader: JsonReader): T? {
    val value = reader.nextInt()
    return (type.enumConstants as Array<T>?)?.firstOrNull { it.value == value }
  }

  companion object Factory : TypeAdapterFactory {
    override fun <T : Any?> create(gson: Gson, type: TypeToken<T>): TypeAdapter<T>? {
      if (type.rawType.isEnum && EnumInt::class.java.isAssignableFrom(type.rawType)) {
        @Suppress("UNCHECKED_CAST")
        return EnumIntTypeAdapter(type.rawType.asSubclass(EnumInt::class.java)) as TypeAdapter<T>
      } else {
        return null
      }
    }
  }
}